from absl import flags

# Where to find data
flags.DEFINE_string('data_path',
                    '/Users/giladram/Documents/NLP/Final Project/unified-summarization/data/finished_files/chunked/train_*',
                    'Path expression to tf.Example datafiles. '
                    'Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
flags.DEFINE_string('model', '', 'must be one of selector/rewriter/end2end')
flags.DEFINE_string('mode', 'train', 'must be one of train/eval/evalall')
flags.DEFINE_boolean('single_pass', False,
                     'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint,'
                     ' i.e. take the current checkpoint, and use it to produce one summary for each example in'
                     ' the dataset, write the summaries to file and then get ROUGE scores for the whole dataset.'
                     ' If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use'
                     ' it to produce summaries for randomly-chosen examples and log the results to screen,'
                     ' indefinitely.')

# Where to save output
flags.DEFINE_integer('max_train_iter', 10000, 'max iterations to train')
flags.DEFINE_integer('save_model_every', 1000, 'save the model every N iterations')
flags.DEFINE_integer('model_max_to_keep', 5, 'save latest N models')
flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '',
                    'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# For eval mode used in rewriter and end2end training
# (This mode will do evaluation during training for choosing best model)
flags.DEFINE_string('eval_method', '',
                    'loss or rouge (loss mode is to get the loss for one batch; rouge mode is to get rouge'
                    ' scores for the whole dataset)')
flags.DEFINE_integer('start_eval_rouge', 30000,
                     'for rouge mode, start evaluating rouge scores after this iteration')

# For evalall mode or (eval mode with eval_method == 'rouge')
flags.DEFINE_string('decode_method', '', 'greedy/beam')
flags.DEFINE_boolean('load_best_eval_model', False, 'evalall mode only')
flags.DEFINE_string('eval_ckpt_path', '', 'evalall mode only, checkpoint path for evalall mode')
flags.DEFINE_boolean('save_pkl', False, 'whether to save the results as pickle files')
flags.DEFINE_boolean('save_vis', False, 'whether to save the results for visualization')

# Load pretrained selector or rewriter
flags.DEFINE_string('pretrained_selector_path', '', 'pretrained selector checkpoint path')
flags.DEFINE_string('pretrained_rewriter_path', '', 'pretrained rewriter checkpoint path')

# For end2end training
flags.DEFINE_float('selector_loss_wt', 5.0, 'weight of selector loss when end2end')
flags.DEFINE_boolean('inconsistent_loss', True, 'whether to minimize inconsistent loss when end2end')
flags.DEFINE_integer('inconsistent_topk', 3, 'choose top K word attention to compute inconsistent loss')

# Hyperparameters for both selector and rewriter
flags.DEFINE_integer('batch_size', 16, 'minibatch size')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
flags.DEFINE_integer('vocab_size', 50000,
                     'Size of vocabulary. These will be read from the vocabulary file in order.'
                     ' If the vocabulary file contains fewer words than this number,'
                     ' or if this number is set to 0, will take all words in the vocabulary file.')
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Hyperparameters for selector only
flags.DEFINE_integer('hidden_dim_selector', 200, 'dimension of RNN hidden states')
flags.DEFINE_integer('max_art_len', 50, 'max timesteps of sentence-level encoder')
flags.DEFINE_integer('max_sent_len', 50, 'max timesteps of word-level encoder')
flags.DEFINE_string('select_method', 'prob', 'prob/ratio/num')
flags.DEFINE_float('thres', 0.4, 'threshold for selecting sentence')
flags.DEFINE_integer('min_select_sent', 5, 'min sentences need to be selected')
flags.DEFINE_integer('max_select_sent', 20, 'max sentences to be selected')
flags.DEFINE_boolean('eval_gt_rouge', False,
                     'whether to evaluate ROUGE scores of ground-truth selected sentences')

# Hyperparameters for rewriter only
flags.DEFINE_integer('hidden_dim_rewriter', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('max_enc_steps', 600, 'max timesteps of encoder (max source text tokens)')
flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
flags.DEFINE_integer('min_dec_steps', 35,
                     'Minimum sequence length of generated summary. Applies only for beam search decoding mode')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', False,
                     'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT'
                     ' coverage until converged, and then train for a short phase WITH coverage afterwards.'
                     ' i.e. to reproduce the results in the ACL paper, turn this off for most of training then'
                     ' turn on for a short phase at the end.')
flags.DEFINE_float('cov_loss_wt', 1.0,
                   'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize'
                   ' coverage loss.')
flags.DEFINE_boolean('convert_to_coverage_model', False,
                     'Convert a non-coverage model to a coverage model. Turn this on and run in train mode.'
                     ' Your current model will be copied to a new version (same name with _cov_init appended)'
                     ' that will be ready to run with coverage flag turned on, for the coverage training stage.')
