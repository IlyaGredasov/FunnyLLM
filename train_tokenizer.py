from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.Train(
    "--input=anecdotes.txt "
    "--model_prefix=sp_anecdote "
    "--vocab_size=8000 "
    "--model_type=unigram "
    "--character_coverage=0.9995"
)
