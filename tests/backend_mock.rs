use npurag::backend::{Backend, Message, MockBackend, Role};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn embed(backend: &MockBackend, text: &str) -> Vec<f32> {
    backend
        .embed(&[text.to_string()])
        .expect("mock never fails")
        .remove(0)
}

#[test]
fn embeddings_are_deterministic_across_instances() {
    let a = MockBackend::new();
    let b = MockBackend::new();
    assert_eq!(
        embed(&a, "backup configuration notes"),
        embed(&b, "backup configuration notes")
    );
}

#[test]
fn embeddings_have_the_declared_dimension() {
    let backend = MockBackend::with_dim(32);
    let vectors = backend
        .embed(&["one".to_string(), "two".to_string()])
        .expect("embeds");
    assert_eq!(vectors.len(), 2);
    assert!(vectors.iter().all(|v| v.len() == 32));
}

#[test]
fn embeddings_are_unit_length() {
    let backend = MockBackend::new();
    for text in [
        "",
        "single",
        "a somewhat longer sentence with several words",
    ] {
        let v = embed(&backend, text);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm} for {text:?}");
    }
}

#[test]
fn shared_words_score_higher_than_unrelated_text() {
    let backend = MockBackend::new();
    let query = embed(&backend, "how did I configure the backup");
    let related = embed(&backend, "the backup is configured to run nightly");
    let unrelated = embed(&backend, "kolejny akapit o zupełnie innym temacie");

    let near = cosine(&query, &related);
    let far = cosine(&query, &unrelated);
    assert!(near > far, "related {near} should beat unrelated {far}");
}

#[test]
fn identical_texts_are_maximally_similar() {
    let backend = MockBackend::new();
    let a = embed(&backend, "identical text");
    let b = embed(&backend, "identical text");
    assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
}

#[test]
fn embedding_is_case_and_punctuation_insensitive() {
    let backend = MockBackend::new();
    let a = embed(&backend, "Backup, notes!");
    let b = embed(&backend, "backup notes");
    assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
}

#[test]
fn an_empty_batch_yields_no_vectors() {
    let backend = MockBackend::new();
    assert!(backend.embed(&[]).expect("embeds").is_empty());
}

#[test]
fn chat_echoes_the_last_user_message() {
    let backend = MockBackend::new();
    let answer = backend
        .chat(
            &[
                Message::system("You answer from context."),
                Message::user("first question"),
                Message::assistant("first answer"),
                Message::user("what did I decide about project X?"),
            ],
            None,
        )
        .expect("chats");

    assert!(
        answer.contains("what did I decide about project X?"),
        "got {answer}"
    );
}

#[test]
fn chat_reports_the_model_override() {
    let backend = MockBackend::new();
    let answer = backend
        .chat(&[Message::user("hi")], Some("some-model"))
        .expect("chats");
    assert!(answer.contains("some-model"), "got {answer}");
}

#[test]
fn chat_without_a_user_message_still_answers() {
    let backend = MockBackend::new();
    let answer = backend
        .chat(&[Message::system("only a system prompt")], None)
        .expect("chats");
    assert!(!answer.is_empty());
}

#[test]
fn the_mock_is_always_healthy() {
    assert!(MockBackend::new().health());
}

#[test]
fn describe_mentions_the_dimension() {
    assert!(MockBackend::with_dim(16).describe().contains("16"));
}

#[test]
fn messages_serialise_in_the_openai_shape() {
    let json = serde_json::to_value(Message::user("hello")).expect("serialises");
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"], "hello");
    assert_eq!(Role::Assistant.as_str(), "assistant");
}
