import importlib


def test_preprocess_dataset_module_imports_from_streammuse_package():
    module = importlib.import_module(
        "streammuse.infrastructure.inference.stanley_stack.preprocess."
        "preprocess_midi2pt_dataset"
    )

    assert module.xf_midi.__package__ == (
        "streammuse.infrastructure.inference.stanley_stack.preprocess"
    )
