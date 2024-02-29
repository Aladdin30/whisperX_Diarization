This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 faster-whisper backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from pyannote-audio (speaker ID labels)
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

Phoneme-Based ASR A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is wav2vec2.0.

Forced Alignment refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

Voice Activity Detection (VAD) is the detection of the presence or absence of human speech.

Speaker Diarization is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.

- This is a Part of output:
  
Speaker: SPEAKER_02

Text:  What do you know about logic?

Speaker: SPEAKER_02

Text: Yeah.

Speaker: SPEAKER_02

Text: Come on.

Speaker: SPEAKER_02

Text: Go ahead.


the origonal repo :https://github.com/m-bain/whisperX
