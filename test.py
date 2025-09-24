from tone import StreamingCTCPipeline, read_audio, read_example_audio

print("creating read_audio object")
# audio = read_example_audio() # or read_audio("your_audio.flac")
# audio = read_audio(r"recordings/rec.mp3")
audio = read_audio(r"recordings/32kHzrec.wav")

# print(read_audio.__doc__)

print("creating pipeline")
pipeline = StreamingCTCPipeline.from_hugging_face()
print("running recognition")
print(pipeline.forward_offline(audio))  # run offline recognition

print('hi!!!!')