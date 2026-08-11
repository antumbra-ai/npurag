use std::collections::HashMap;
use std::path::PathBuf;

use npurag::config::{BackendPreset, Config, Overrides, AMD_FLM, INTEL_OVMS};

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |key: &str| map.get(key).cloned()
}

#[test]
fn defaults_ship_both_backend_presets() {
    let config = Config::default();
    assert_eq!(config.backend, AMD_FLM);

    let amd = config.backends.get(AMD_FLM).expect("amd preset");
    assert_eq!(amd.base_url, "http://localhost:52625/v1");
    assert_eq!(amd.embed_model, "embeddinggemma-300m");

    let intel = config.backends.get(INTEL_OVMS).expect("intel preset");
    // OVMS exposes /v3, and the whole portability story depends on that prefix
    // living in configuration rather than in the HTTP client.
    assert!(intel.base_url.ends_with("/v3"), "got {}", intel.base_url);
}

#[test]
fn resolving_picks_the_active_preset() {
    let config = Config {
        backend: INTEL_OVMS.to_string(),
        ..Default::default()
    };

    let resolved = config.resolve_backend().expect("resolves");
    assert_eq!(resolved.name, INTEL_OVMS);
    assert_eq!(resolved.base_url, "http://localhost:8000/v3");
}

#[test]
fn resolving_an_unknown_backend_names_the_known_ones() {
    let config = Config {
        backend: "nope".to_string(),
        ..Default::default()
    };

    let err = config.resolve_backend().unwrap_err().to_string();
    assert!(err.contains("nope"), "got {err}");
    assert!(err.contains(AMD_FLM), "got {err}");
    assert!(err.contains(INTEL_OVMS), "got {err}");
}

#[test]
fn a_trailing_slash_in_base_url_is_normalised_away() {
    let mut config = Config::default();
    config.apply_overrides(&Overrides {
        base_url: Some("http://localhost:9000/v1/".to_string()),
        ..Default::default()
    });

    assert_eq!(
        config.resolve_backend().unwrap().base_url,
        "http://localhost:9000/v1"
    );
}

#[test]
fn round_trips_through_toml() {
    let config = Config::default();
    let text = config.to_toml().expect("serialises");
    let parsed = Config::from_toml(&text).expect("parses");
    assert_eq!(config, parsed);
}

#[test]
fn a_partial_config_file_keeps_the_defaults() {
    let parsed = Config::from_toml("chunk_tokens = 256\n").expect("parses");
    assert_eq!(parsed.chunk_tokens, 256);
    assert_eq!(parsed.backend, AMD_FLM);
    assert_eq!(parsed.max_file_size_mb, 5);
    assert!(parsed.backends.contains_key(INTEL_OVMS));
}

#[test]
fn a_config_file_can_add_its_own_preset() {
    let text = r#"
backend = "custom"

[backends.custom]
base_url = "http://192.168.0.10:9999/v1"
embed_model = "my-embed"
chat_model = "my-chat"
"#;
    let parsed = Config::from_toml(text).expect("parses");
    let resolved = parsed.resolve_backend().expect("resolves");
    assert_eq!(resolved.name, "custom");
    assert_eq!(resolved.embed_model, "my-embed");
}

#[test]
fn env_overrides_the_file() {
    let mut config = Config::default();
    config.apply_env_with(env(&[
        ("NPURAG_BACKEND", INTEL_OVMS),
        ("NPURAG_BASE_URL", "http://otherhost:8000/v1"),
        ("NPURAG_CHAT_MODEL", "some-llm-int4-ov"),
    ]));

    let resolved = config.resolve_backend().expect("resolves");
    assert_eq!(resolved.name, INTEL_OVMS);
    assert_eq!(resolved.base_url, "http://otherhost:8000/v1");
    assert_eq!(resolved.chat_model, "some-llm-int4-ov");
    // Untouched fields keep the preset's value.
    assert_eq!(resolved.embed_model, "embeddinggemma-300m");
}

#[test]
fn cli_overrides_win_over_env() {
    let mut config = Config::default();
    config.apply_env_with(env(&[("NPURAG_BASE_URL", "http://from-env:1111/v1")]));
    config.apply_overrides(&Overrides {
        base_url: Some("http://from-cli:2222/v1".to_string()),
        ..Default::default()
    });

    assert_eq!(
        config.resolve_backend().unwrap().base_url,
        "http://from-cli:2222/v1"
    );
}

#[test]
fn overriding_the_url_of_an_unknown_backend_creates_that_preset() {
    let mut config = Config::default();
    config.apply_overrides(&Overrides {
        backend: Some("bespoke".to_string()),
        base_url: Some("http://localhost:1234/v2".to_string()),
        ..Default::default()
    });

    let resolved = config.resolve_backend().expect("resolves");
    assert_eq!(resolved.name, "bespoke");
    assert_eq!(resolved.base_url, "http://localhost:1234/v2");
}

#[test]
fn a_preset_without_a_base_url_is_rejected() {
    let mut config = Config {
        backend: "empty".to_string(),
        ..Default::default()
    };
    config.backends.insert(
        "empty".to_string(),
        BackendPreset {
            base_url: "  ".to_string(),
            embed_model: "e".to_string(),
            chat_model: "c".to_string(),
            rerank_model: None,
        },
    );

    let err = config.resolve_backend().unwrap_err().to_string();
    assert!(err.contains("base_url"), "got {err}");
}

#[test]
fn an_explicit_db_path_beats_the_per_root_default() {
    let mut config = Config::default();
    let root = PathBuf::from("/tmp/somewhere");
    let default_path = config.db_path_for(&root).expect("default path");
    assert!(default_path.ends_with("index.db"), "got {default_path:?}");

    config.apply_overrides(&Overrides {
        db: Some(PathBuf::from("/tmp/explicit.db")),
        ..Default::default()
    });
    assert_eq!(
        config.db_path_for(&root).unwrap(),
        PathBuf::from("/tmp/explicit.db")
    );
}

#[test]
fn different_roots_get_different_index_paths() {
    let config = Config::default();
    let a = config.db_path_for(&PathBuf::from("/home/u/notes")).unwrap();
    let b = config.db_path_for(&PathBuf::from("/home/u/code")).unwrap();
    assert_ne!(a, b);
}
