import os
import urllib.request

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_audio")

# Public domain / CC0 test files
FILES = {
    "sample_male.wav": "https://www2.cs.uic.edu/~i101/SoundFiles/MaleSpeech-16k.wav",
    "sample_female.wav": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav"
}

def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    for filename, url in FILES.items():
        out_path = os.path.join(SAMPLES_DIR, filename)
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved to {out_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
