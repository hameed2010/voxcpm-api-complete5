def test_project_files_exist():
    from pathlib import Path
    assert Path("app/main.py").exists()
    assert Path("app/core/model_manager.py").exists()
    assert Path("app/services/voice/clone_service.py").exists()
