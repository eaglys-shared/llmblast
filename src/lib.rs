use pyo3::prelude::*;
use serde_json::json;
use std::sync::Arc;
use std::sync::OnceLock;

const API_OPENAI: &str = "https://api.openai.com/v1/chat/completions";

#[pyclass]
#[derive(Debug, Clone)]
pub enum Provider {
    OpenAI { model_name: String, api_key: String },
    Anthropic { model_name: String, api_key: String },
}

// Define a static HTTP client
static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

// Helper function to get or initialize the HTTP client
fn get_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client")
    })
}

pub async fn _call_llm(prompt: String, provider: Provider) -> anyhow::Result<String> {
    let (api_url, headers, api_key) = match &provider {
        Provider::OpenAI {
            model_name,
            api_key,
        } => {
            let api_url = API_OPENAI;
            let headers = json!({
                "model": model_name,
                "temperature": 0.0,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                }]
            });
            (api_url, headers, api_key)
        }
        Provider::Anthropic { .. } => {
            unimplemented!()
        }
    };
    
    // Use the static client instead of creating a new one
    let res = get_http_client()
        .post(api_url)
        .header("Content-Type", "application/json")
        .header("authorization", format!("Bearer {}", api_key))
        .json(&headers)
        .send()
        .await?
        .text()
        .await?;

    // Extract the content based on the provider
    let json_response: serde_json::Value = serde_json::from_str(&res)?;

    let content = match &provider {
        Provider::OpenAI { .. } => {
            // For OpenAI: choices[0].message.content
            json_response
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str())
                .ok_or_else(|| anyhow::anyhow!("Failed to extract content from OpenAI response"))?
                .to_string()
        }
        Provider::Anthropic { .. } => {
            // For Anthropic: content[0].text
            json_response
                .get("content")
                .and_then(|content| content.get(0))
                .and_then(|block| block.get("text"))
                .and_then(|text| text.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to extract content from Anthropic response")
                })?
                .to_string()
        }
    };
    Ok(content)
}

pub async fn _call_llm_batch(
    prompts: Vec<String>,
    provider: Provider,
) -> anyhow::Result<Vec<String>> {
    let provider = Arc::new(provider);
    let mut tasks = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let provider_clone: Arc<Provider> = Arc::clone(&provider);
        let prompt: Arc<String> = Arc::new(prompt.clone());
        let task = tokio::task::spawn(async move {
            _call_llm((*prompt).clone(), (*provider_clone).clone()).await
        });
        tasks.push(task);
    }

    let results = futures::future::join_all(tasks).await;

    let mut responses = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(Ok(response)) => responses.push(response),
            Ok(Err(e)) => return Err(e),
            Err(e) => anyhow::bail!("Task join error: {}", e),
        }
    }

    Ok(responses)
}

#[pyfunction]
fn call_llm_batch(prompt: Vec<String>, provider: Provider) -> anyhow::Result<Vec<String>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { _call_llm_batch(prompt, provider).await })
}

// A Python module implemented in Rust.
#[pymodule]
fn llmblast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Provider>()?;
    m.add_function(wrap_pyfunction!(call_llm_batch, m)?)?;
    Ok(())
}