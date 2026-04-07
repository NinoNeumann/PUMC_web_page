import json
import math
import os

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .utils.model_loader import ModelService

_CUTOFFS_CACHE = None


def _load_lp_risk_cutoffs():
    """lp 分组阈值：与 DeepLearning_Risk_Groups.csv 中三分位结果一致。"""
    global _CUTOFFS_CACHE
    if _CUTOFFS_CACHE is not None:
        return _CUTOFFS_CACHE
    path = os.path.join(
        settings.BASE_DIR, 'predictor', 'model_files', 'risk_group_cutoffs.json'
    )
    if not os.path.isfile(path):
        _CUTOFFS_CACHE = {
            'lp_low_max': -2.343692,
            'lp_mid_max': -0.34264246,
            'lp_display_min': -3.0107012,
            'lp_display_max': 1.2592012,
        }
        return _CUTOFFS_CACHE
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    _CUTOFFS_CACHE = {
        'lp_low_max': float(data['lp_low_max']),
        'lp_mid_max': float(data['lp_mid_max']),
        'lp_display_min': float(data['lp_display_min']),
        'lp_display_max': float(data['lp_display_max']),
    }
    return _CUTOFFS_CACHE


def _classify_lp_risk_group(lp, cutoffs):
    """
    与 CSV 规则一致：
    Low: lp <= lp_low_max；Intermediate: lp_low_max < lp <= lp_mid_max；High: lp > lp_mid_max
    """
    t1, t2 = cutoffs['lp_low_max'], cutoffs['lp_mid_max']
    if lp <= t1:
        return 'Low Risk', 'low'
    if lp <= t2:
        return 'Intermediate Risk', 'intermediate'
    return 'High Risk', 'high'


def _lp_to_bar_percent(lp, cutoffs):
    lo, hi = cutoffs['lp_display_min'], cutoffs['lp_display_max']
    span = hi - lo
    if span <= 0:
        return 50.0
    pct = (lp - lo) / span * 100.0
    return min(max(pct, 1.0), 99.0)


FALLBACK_OPTIONS = {
    'Gender': [0, 1],
    'Antibody': [0, 1],
    'Coronary_Heart_Disease': [0, 1],
    'interstitial_lung_disease': [0, 1],
    'Hematuria': [0, 1],
    'Musculoskeletal_Involvement': [0, 1],
    'Pulmonary_Involvement': [0, 1],
    'Cr_category': [0, 1, 2, 3],
    'ESR_category': [0, 1, 2, 3],
    'PLT_category': [0, 1, 2, 3],
    'Smoking_History': [0, 1]
}

DEFAULT_VALUES = {
    'Age': 55,
    'Onset_Age': 50,
    'Hb': 130,
    'WBC': 6.5,
    'ALT': 25,
    'CRP': 5.0,
    'Gender': 0,
    'Antibody': 0,
    'Coronary_Heart_Disease': 0,
    'interstitial_lung_disease': 0,
    'Hematuria': 0,
    'Musculoskeletal_Involvement': 0,
    'Pulmonary_Involvement': 0,
    'Cr_category': 0,
    'ESR_category': 0,
    'PLT_category': 0,
    'Smoking_History': 0
}

SLIDER_CONFIG = {
    'Age': {'min': 18, 'max': 90, 'step': 1},
    'Onset_Age': {'min': 18, 'max': 90, 'step': 1},
    'Hb': {'min': 40, 'max': 180, 'step': 1},
    'WBC': {'min': 0, 'max': 100, 'step': 0.1},
    'ALT': {'min': 0, 'max': 500, 'step': 1},
    'CRP': {'min': 0, 'max': 200, 'step': 0.1},
}


def _align_categorical_value(value, options):
    if not options:
        return value
    for o in options:
        if str(o) == str(value):
            return o
    return value


def _build_form_fields(service, post=None):
    """post: QueryDict from request.POST to repopulate after submit; None for defaults."""
    fields = []
    post = post or {}

    for col in service.numeric_cols:
        raw = post.get(col)
        if raw is not None and str(raw).strip() != '':
            try:
                default_val = float(raw)
            except ValueError:
                default_val = DEFAULT_VALUES.get(col, 0)
        else:
            default_val = DEFAULT_VALUES.get(col, 0)

        field_config = {
            'name': col,
            'label': col,
            'value': default_val
        }
        if col in SLIDER_CONFIG:
            field_config.update({
                'type': 'range',
                'min': SLIDER_CONFIG[col]['min'],
                'max': SLIDER_CONFIG[col]['max'],
                'step': SLIDER_CONFIG[col]['step']
            })
        else:
            field_config.update({
                'type': 'number',
                'step': 'any'
            })
        fields.append(field_config)

    for col in service.categorical_cols:
        options = []
        if service.encoders and col in service.encoders:
            options = list(service.encoders[col].classes_)
        elif col in FALLBACK_OPTIONS:
            options = FALLBACK_OPTIONS[col]

        field_type = 'select' if options else 'text'
        if not options and isinstance(DEFAULT_VALUES.get(col), (int, float)):
            field_type = 'number'

        raw = post.get(col)
        if raw is not None and str(raw).strip() != '':
            default_val = _align_categorical_value(raw, options)
        else:
            default_val = _align_categorical_value(DEFAULT_VALUES.get(col, ""), options)

        fields.append({
            'name': col,
            'type': field_type,
            'label': col,
            'options': options,
            'value': default_val
        })

    return fields


@require_http_methods(['GET', 'POST'])
def index(request):
    service = ModelService.get_instance()

    if not service.config:
        return render(request, 'predictor/index.html', {
            'error': 'Model not loaded or config file missing.',
            'fields': [],
        })

    if request.method == 'POST':
        if service.numeric_cols is None:
            return render(request, 'predictor/index.html', {
                'error': 'Model not loaded. Please ensure model files are placed in predictor/model_files/.',
                'fields': _build_form_fields(service, request.POST),
            })

        input_data = {}
        for col in service.numeric_cols:
            val = request.POST.get(col)
            if val:
                input_data[col] = float(val)
            else:
                input_data[col] = 0.0

        for col in service.categorical_cols:
            val = request.POST.get(col)
            if val:
                if col in FALLBACK_OPTIONS:
                    try:
                        input_data[col] = int(val)
                    except ValueError:
                        input_data[col] = val
                else:
                    input_data[col] = val
            else:
                input_data[col] = 0

        result = service.predict(input_data)

        if 'risk_score' in result:
            try:
                lp = float(result['risk_score'])
                result['lp'] = lp
                cutoffs = _load_lp_risk_cutoffs()
                label, band = _classify_lp_risk_group(lp, cutoffs)
                result['risk_group'] = label
                result['risk_band'] = band
                result['percentile'] = _lp_to_bar_percent(lp, cutoffs)

                base_survival_1yr = 0.90
                base_survival_2yr = 0.85
                hazard_ratio = math.exp(lp)
                result['survival_1yr'] = round(base_survival_1yr ** hazard_ratio * 100, 1)
                result['survival_2yr'] = round(base_survival_2yr ** hazard_ratio * 100, 1)

            except Exception as e:
                print(f"Error deriving risk group from lp: {e}")
                result['percentile'] = 50
                result['risk_group'] = 'Intermediate Risk'
                result['risk_band'] = 'intermediate'

        return render(request, 'predictor/index.html', {
            'fields': _build_form_fields(service, request.POST),
            'result': result,
            'inputs': input_data,
        })

    return render(request, 'predictor/index.html', {
        'fields': _build_form_fields(service, None),
    })
