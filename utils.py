import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import functools
import time

from config import PLOT_SETTINGS, FILE_SETTINGS

# =============================================================================
# 로깅 설정
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger("TurbineAnalyzer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()

# =============================================================================
# 데코레이터
# =============================================================================

def timer(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

def error_handler(func):
    """예외 처리 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__}에서 오류 발생: {str(e)}")
            raise
    return wrapper

# =============================================================================
# 시각화 설정
# =============================================================================

def setup_plot_style():
    """matplotlib 및 seaborn 스타일 설정"""
    try:
        # 한글 폰트 설정
        plt.rcParams['font.family'] = PLOT_SETTINGS['font_family']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 기본 스타일 설정
        plt.style.use('default')  # seaborn 스타일이 deprecated되어 default 사용
        sns.set_palette(PLOT_SETTINGS['color_palette'])
        
        # 기본 figure 크기 설정
        plt.rcParams['figure.figsize'] = PLOT_SETTINGS['figure_size']
        plt.rcParams['figure.dpi'] = PLOT_SETTINGS['dpi']
        
        logger.info("시각화 스타일 설정 완료")
        
    except Exception as e:
        logger.warning(f"폰트 설정 실패: {e}. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans'

def create_figure(figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """표준 figure 생성"""
    if figsize is None:
        figsize = PLOT_SETTINGS['figure_size']
    
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

# =============================================================================
# 데이터 처리 유틸리티
# =============================================================================

@error_handler
def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """안전한 데이터 로딩"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")
    
    logger.info(f"데이터 로딩 완료: {df.shape[0]}행, {df.shape[1]}열")
    return df

def validate_turbine_data(df: pd.DataFrame) -> Dict[str, Any]:
    """풍력 터빈 데이터 검증"""
    required_columns = [
        'Time', 'Power', 'windspeed_100m', 'winddirection_100m', 
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m'
    ]
    
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'data_quality': {},
        'summary': {}
    }
    
    # 필수 컬럼 확인
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation_result['is_valid'] = False
        validation_result['missing_columns'] = missing_cols
    
    # 데이터 품질 확인
    if validation_result['is_valid']:
        for col in required_columns:
            if col in df.columns:
                validation_result['data_quality'][col] = {
                    'null_count': df[col].isnull().sum(),
                    'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'data_type': str(df[col].dtype)
                }
        
        # 기본 통계
        validation_result['summary'] = {
            'total_rows': len(df),
            'date_range': None,
            'power_range': None
        }
        
        # 시간 데이터 확인
        if 'Time' in df.columns:
            try:
                time_series = pd.to_datetime(df['Time'])
                validation_result['summary']['date_range'] = {
                    'start': time_series.min(),
                    'end': time_series.max(),
                    'duration_days': (time_series.max() - time_series.min()).days
                }
            except:
                pass
        
        # 발전량 범위 확인
        if 'Power' in df.columns:
            validation_result['summary']['power_range'] = {
                'min': df['Power'].min(),
                'max': df['Power'].max(),
                'mean': df['Power'].mean()
            }
    
    return validation_result

def preprocess_turbine_data(df: pd.DataFrame) -> pd.DataFrame:
    """풍력 터빈 데이터 전처리"""
    df_processed = df.copy()
    
    # 시간 데이터 처리
    if 'Time' in df_processed.columns:
        df_processed['Time'] = pd.to_datetime(df_processed['Time'])
        df_processed.set_index('Time', inplace=True)
    
    # 수치형 데이터 타입 확인 및 변환
    numeric_columns = [
        'Power', 'windspeed_100m', 'winddirection_100m', 
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m'
    ]
    
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # 기본적인 이상치 제거 (선택적)
    # 음수 발전량 제거
    if 'Power' in df_processed.columns:
        df_processed = df_processed[df_processed['Power'] >= 0]
    
    # 비현실적인 풍속 제거 (0~50 m/s 범위)
    if 'windspeed_100m' in df_processed.columns:
        df_processed = df_processed[
            (df_processed['windspeed_100m'] >= 0) & 
            (df_processed['windspeed_100m'] <= 50)
        ]
    
    logger.info(f"전처리 완료: {len(df_processed)}행 남음")
    return df_processed

# =============================================================================
# 통계 및 계산 유틸리티
# =============================================================================

def calculate_statistics(series: pd.Series) -> Dict[str, float]:
    """기본 통계 계산"""
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q50': series.quantile(0.50),
        'q75': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    return stats

def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
    """이상치 탐지"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > 3
    
    else:
        raise ValueError("지원하는 방법: 'iqr', 'zscore'")

def calculate_efficiency_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """효율성 지표 계산"""
    efficiency_ratio = actual / predicted
    
    metrics = {
        'mean_efficiency': efficiency_ratio.mean(),
        'efficiency_std': efficiency_ratio.std(),
        'efficiency_min': efficiency_ratio.min(),
        'efficiency_max': efficiency_ratio.max(),
        'low_efficiency_ratio': (efficiency_ratio < 0.8).sum() / len(efficiency_ratio)
    }
    return metrics

# =============================================================================
# 파일 및 경로 유틸리티
# =============================================================================

def ensure_directory(path: str) -> None:
    """디렉토리가 없으면 생성"""
    os.makedirs(path, exist_ok=True)

def get_safe_filename(filename: str) -> str:
    """안전한 파일명 생성"""
    import re
    # 특수문자 제거 및 공백을 underscore로 변경
    safe_name = re.sub(r'[^\w\s-]', '', filename)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name

def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """DataFrame 안전하게 저장"""
    ensure_directory(os.path.dirname(filepath))
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        df.to_csv(filepath, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df.to_excel(filepath, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
    
    logger.info(f"파일 저장 완료: {filepath}")

# =============================================================================
# 날짜/시간 유틸리티
# =============================================================================

def get_time_features(timestamp: pd.Timestamp) -> Dict[str, int]:
    """시간 특성 추출"""
    return {
        'year': timestamp.year,
        'month': timestamp.month,
        'day': timestamp.day,
        'hour': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'day_of_year': timestamp.dayofyear,
        'week_of_year': timestamp.isocalendar()[1],
        'is_weekend': timestamp.dayofweek >= 5
    }

def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 형태로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        return f"{seconds/60:.1f}분"
    else:
        return f"{seconds/3600:.1f}시간"

# =============================================================================
# 초기화 함수
# =============================================================================

def initialize_utils():
    """유틸리티 모듈 초기화"""
    setup_plot_style()
    logger.info("Utils 모듈 초기화 완료")

# 모듈 로드시 자동 실행
initialize_utils()