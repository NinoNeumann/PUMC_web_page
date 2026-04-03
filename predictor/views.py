import math
from django.shortcuts import render
from django.http import JsonResponse
from .utils.model_loader import ModelService

# Manual fallback options for fields that might be numeric in source but treated as categorical
FALLBACK_OPTIONS = {
    '性别Code': [0, 1],
    '五因子评分': [0, 1, 2, 3, 4, 5],
    '是否使用传统DMARDs药物': [0, 1],
    '肺纤维化': [0, 1],
    '肾功能': [0, 1],
    '鼻痂溃疡': [0, 1],
    '尿红细胞': [0, 1],
    '肺': [0, 1],
    '关节痛': [0, 1],
    'ANA': [0, 1],
    '新诊断疾病': [0, 1],
    '有无并发肿瘤': [0, 1],
    # Guessing binary for others. If ESR/Cr category are codes 0/1/2..
    'ESR_category': [0, 1, 2, 3], 
    'Cr_category': [0, 1, 2, 3]
}

DEFAULT_VALUES = {
    '年龄': 55,
    '发病年龄': 50,
    'Hb': 130,
    'WBC': 6.5,
    'PLT': 220,
    'CRP': 5.0,
    '性别Code': 0,
    '五因子评分': 0,
    '是否使用传统DMARDs药物': 0,
    '肺纤维化': 0,
    '肾功能': 0,
    '鼻痂溃疡': 0,
    '尿红细胞': 0,
    '肺': 0,
    '关节痛': 0,
    'ANA': 0,
    '新诊断疾病': 0,
    '有无并发肿瘤': 0,
    'ESR_category': 0,
    'Cr_category': 0
}

# Configuration for slider fields
SLIDER_CONFIG = {
    '年龄': {'min': 18, 'max': 90, 'step': 1},
    '发病年龄': {'min': 18, 'max': 90, 'step': 1},
    'Hb': {'min': 40, 'max': 180, 'step': 1},
    'WBC': {'min': 0, 'max': 100, 'step': 0.1},
    'PLT': {'min': 0, 'max': 1000, 'step': 1},
    'CRP': {'min': 0, 'max': 200, 'step': 0.1},
    '五因子评分': {'min': 0, 'max': 5, 'step': 1},
}

def index(request):
    service = ModelService.get_instance()
    
    if not service.config:
        return render(request, 'predictor/index.html', {
            'error': 'Model not loaded or config file missing.'
        })

    # Prepare fields for the form
    # We differentiate between numeric and categorical to render appropriate inputs
    fields = []
    
    # Numeric fields
    for col in service.numeric_cols:
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
        
    # Categorical fields - we need options if possible
    # The encoders have classes_. 
    for col in service.categorical_cols:
        options = []
        is_fallback = False
        
        if service.encoders and col in service.encoders:
            options = list(service.encoders[col].classes_)
        elif col in FALLBACK_OPTIONS:
             options = FALLBACK_OPTIONS[col]
             is_fallback = True
        
        # If still no options, maybe fallback to number input or text input?
        field_type = 'select' if options else 'text'
        if not options:
            # Try to see if default value is a number, use number input
            if isinstance(DEFAULT_VALUES.get(col), (int, float)):
                 field_type = 'number'
        
        default_val = DEFAULT_VALUES.get(col, "")
        if options and default_val not in options and len(options) > 0:
             # default_val = options[0] # Don't force override if not in list, let template handle select
             pass

        fields.append({
            'name': col,
            'type': field_type,
            'label': col,
            'options': options,
            'value': default_val
        })
        
    return render(request, 'predictor/index.html', {'fields': fields})

def predict(request):
    if request.method == 'POST':
        service = ModelService.get_instance()
        
        if not service.config or service.numeric_cols is None:
             return render(request, 'predictor/index.html', {
                'error': 'Model not loaded. Please ensure model files are placed in predictor/model_files/.'
            })
        
        # Collect data from POST
        input_data = {}
        
        for col in service.numeric_cols:
            val = request.POST.get(col)
            if val:
                input_data[col] = float(val)
            else:
                 input_data[col] = 0.0 # Default?
                 
        for col in service.categorical_cols:
            val = request.POST.get(col)
            if val:
                # If encoder exists, we might need original value (str/int)
                # If fallback (numeric options), convert to int/float if possible
                if col in FALLBACK_OPTIONS:
                    try:
                        input_data[col] = int(val)
                    except ValueError:
                        input_data[col] = val
                else:
                    input_data[col] = val
            else:
                input_data[col] = 0 # Default fallback
                
        result = service.predict(input_data)
        
        # Calculate percentile if risk score is available
        if 'risk_score' in result:
            try:
                # Assuming risk score is log-hazard ratio, roughly normally distributed centered at 0
                # Map to percentile (0-100)
                z_score = result['risk_score']
                # Using a scaling factor? If the training data std dev was 1, then this is fine.
                # If we don't know, we assume standard normal for the "Risk Score" interpretation.
                percentile = 0.5 * (1 + math.erf(z_score / math.sqrt(2))) * 100
                result['percentile'] = min(max(percentile, 1), 99) # Clamp between 1 and 99
                
                # Mock survival probabilities if not provided by model
                # This is a placeholder since we can't calculate true survival without baseline hazard
                # Using a logistic function to mock survival decrease with risk
                base_survival_1yr = 0.90
                base_survival_2yr = 0.85
                
                # Higher risk score -> Lower survival
                # survival = base ^ exp(risk_score)
                hazard_ratio = math.exp(z_score)
                result['survival_1yr'] = round(base_survival_1yr ** hazard_ratio * 100, 1)
                result['survival_2yr'] = round(base_survival_2yr ** hazard_ratio * 100, 1)
                
            except Exception as e:
                print(f"Error calculating percentile: {e}")
                result['percentile'] = 50

        return render(request, 'predictor/result.html', {'result': result, 'inputs': input_data})
        
    return JsonResponse({'error': 'Method not allowed'}, status=405)
