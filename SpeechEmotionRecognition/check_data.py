import os

data_dir = 'data/RAVDESS'  # Update if dataset is elsewhere, e.g., 'C:/Users/moham/Downloads/RAVDESS'
print("Checking directory:", os.path.abspath(data_dir))
if os.path.exists(data_dir):
    print("Directory exists. Contents:", os.listdir(data_dir))
    for actor_dir in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_dir)
        if os.path.isdir(actor_path):
            wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            print(f"Actor {actor_dir}: {len(wav_files)} .wav files")
            if len(wav_files) > 0:
                print("Sample file:", wav_files[0])
else:
    print("Directory not found! Please check the path.")