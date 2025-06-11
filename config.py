import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# =============================================================================
# LLM 제공자 설정
# =============================================================================

# 현재 사용할 LLM 제공자 선택
LLM_PROVIDER = "gemini"  # "gemini", "openai", "claude", "local" 등

# 각 LLM 제공자별 설정
LLM_CONFIGS = {
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_name": "gemini-1.5-flash",  # 또는 gemini-1.5-pro
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.8,
        "top_k": 40,
    },
    
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-4o-mini",  # 또는 gpt-4o
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.8,
    },
    
    "claude": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.8,
    },
    
    # 추후 로컬 모델이나 다른 제공자 추가 가능
    "local": {
        "model_path": "path/to/local/model",
        "temperature": 0.3,
        "max_tokens": 2048,
    }
}

# =============================================================================
# 데이터 분석 설정
# =============================================================================

# 풍력 터빈 분석 관련 설정
TURBINE_ANALYSIS = {
    "cut_in_speed": 3.5,  # 터빈 가동 최소 풍속 (m/s)
    "cut_out_speed": 25.0,  # 터빈 정지 최대 풍속 (m/s)
    "wind_speed_bin_size": 0.5,  # 풍속 구간 크기
    "inefficiency_threshold_factor": 2.0,  # 비효율 판정 기준 (평균 - n*std)
}

# 머신러닝 모델 설정
ML_SETTINGS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "models": {
        "random_forest": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1
        }
    }
}

# =============================================================================
# 앱 전반적인 설정
# =============================================================================

# Streamlit 앱 설정
APP_CONFIG = {
    "page_title": "AI 설비 성능 분석가",
    "page_icon": "🔧",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 파일 업로드 설정
FILE_SETTINGS = {
    "max_file_size_mb": 200,
    "allowed_extensions": [".csv", ".xlsx", ".xls"],
    "default_data_path": "data/Location1.csv"
}

# 시각화 설정
PLOT_SETTINGS = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "font_family": "Malgun Gothic",  # 한글 폰트
    "color_palette": "viridis"
}

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_current_llm_config():
    """현재 선택된 LLM 제공자의 설정을 반환"""
    if LLM_PROVIDER not in LLM_CONFIGS:
        raise ValueError(f"지원하지 않는 LLM 제공자입니다: {LLM_PROVIDER}")
    
    config = LLM_CONFIGS[LLM_PROVIDER].copy()
    
    # API 키 검증
    if "api_key" in config and not config["api_key"]:
        raise ValueError(f"{LLM_PROVIDER} API 키가 설정되지 않았습니다. 환경변수를 확인해주세요.")
    
    return config

def switch_llm_provider(provider_name):
    """LLM 제공자를 동적으로 변경"""
    global LLM_PROVIDER
    if provider_name in LLM_CONFIGS:
        LLM_PROVIDER = provider_name
        return True
    return False

def get_available_providers():
    """사용 가능한 LLM 제공자 목록 반환"""
    available = []
    for provider, config in LLM_CONFIGS.items():
        if config.get("api_key") or provider == "local":
            available.append(provider)
    return available

def validate_config():
    """설정 검증"""
    errors = []
    
    # LLM 설정 검증
    try:
        get_current_llm_config()
    except ValueError as e:
        errors.append(f"LLM 설정 오류: {e}")
    
    # 데이터 파일 존재 여부 확인
    if not os.path.exists(FILE_SETTINGS["default_data_path"]):
        errors.append(f"기본 데이터 파일이 없습니다: {FILE_SETTINGS['default_data_path']}")
    
    return errors

if __name__ == "__main__":
    # 설정 테스트
    print("=== 설정 검증 ===")
    errors = validate_config()
    if errors:
        for error in errors:
            print(f"❌ {error}")
    else:
        print("✅ 모든 설정이 올바릅니다.")
    
    print(f"\n현재 LLM 제공자: {LLM_PROVIDER}")
    print(f"사용 가능한 제공자: {get_available_providers()}")