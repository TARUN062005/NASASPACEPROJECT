# inspect_preprocessor.py
import joblib
import os

print("🔍 Inspecting preprocessor structure...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    if os.path.exists('koi_preprocessor.pkl'):
        print("\n📦 Loading koi_preprocessor.pkl...")
        data = joblib.load('koi_preprocessor.pkl')
        print(f"✅ Loaded successfully!")
        print(f"Data type: {type(data)}")
        print(f"Keys: {data.keys() if hasattr(data, 'keys') else 'Not a dict'}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n📋 Key: {key}")
                print(f"   Type: {type(value)}")
                if key == 'preprocessor':
                    print(f"   Has preprocessor: {value is not None}")
                    if hasattr(value, 'selected_features'):
                        print(f"   Selected features: {value.selected_features}")
                elif key == 'label_encoder':
                    print(f"   Has label_encoder: {value is not None}")
                    if hasattr(value, 'classes_'):
                        print(f"   Classes: {value.classes_}")
                else:
                    print(f"   Value preview: {str(value)[:100]}...")
        else:
            print(f"\n📋 Direct object attributes:")
            for attr in dir(data):
                if not attr.startswith('_'):
                    try:
                        val = getattr(data, attr)
                        print(f"   {attr}: {type(val)}")
                    except:
                        pass
    else:
        print("❌ koi_preprocessor.pkl not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    