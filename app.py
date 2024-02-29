import whisperx
import torch

print(torch.cuda.is_available())  # to check if cude available or not will return True is available
device = "cuda"
batch_size = 4 # reduce if low on GPU mem
compute_type = "float16"

audio_file = "/content/test_Di4.wav"

model = whisperx.load_model("large-v3", device, compute_type=compute_type)
def WisperX(audio_file):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    return result,audio



def diarizer(result,audio):
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",
                                                    device=device)

    diarize_segments = diarize_model(audio, min_speakers=3, max_speakers=3)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"])
    print(result)


    # Extract speaker and text from each segment
    speaker_text_pairs = [(segment["speaker"], segment["text"]) for segment in result["segments"]]

    # Print the result
    for speaker, text in speaker_text_pairs:
        print("Speaker:", speaker)
        print("Text:", text)
    return speaker_text_pairs

def all(audio_file):
    result,audio=WisperX(audio_file)
    end=diarizer(result,audio)

    return end
the_final_output=all(audio_file)
print(the_final_output)






