import argparse
import numpy as np
import torch
import torch.optim as optim
from data_factory.dataloader import *
from arg_completion import *
from models.flow.invertible_net import *
# from generative_classifier import generative_classifier
from data_factory.dataloader import IMAGE_LOADOR
from pl_bolts.models.self_supervised import CPCV2
import utils
import argparse
import pickle
import numpy as np
import torch
import torch.optim as optim
import logging
from discriminative_classifier import discriminative_classifier
from trainers import att_classifier_trainer
from utils_new import create_filename, create_modelname, config_process, fix_act_norm_issue, make_training_kwargs, read_attribute
import copy
from inn_vae import inn_vae
from training import callbacks
logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser(description='ZSL')

    parser.add_argument('--cfg_file',
                        type=str,
                        dest='cfg_file',
                        default='configs/ResNet101_AwA2_SS_C.yaml')

    # data loader params
    parser.add_argument('--data_root',
                        default='../../../media/data/',
                        type=str,
                        metavar='DIRECTORY',
                        help='path to data directory')
    parser.add_argument('--dataset',
                        type=str,
                        default='AwA2',
                        choices=[],
                        help='which dataset to use')
    parser.add_argument("--dir",
                        type=str,
                        default=".",
                        help="Base directory of repo")
    parser.add_argument('--zsl_type',
                        type=str,
                        default='conventional',
                        choices=[],
                        help='')
    parser.add_argument('--split',
                        default='standard_split',
                        metavar='NAME',
                        help='image dir for loading images')
    parser.add_argument('--validation',
                        action='store_true',
                        default=False,
                        help='enable cross validation mode')
    # experimental results
    parser.add_argument('--result_root',
                        type=str,
                        default='./results',
                        help='location of the results')
    parser.add_argument('--save',
                        type=str,
                        default='exp',
                        help='id used for storing intermediate results')

    parser.add_argument("--debug",
                        action="store_true",
                        default=True,
                        help="Debug mode (more log output, additional callbacks)")

    parser.add_argument("--pretraining_att_classifier",
                        action="store_true",
                        default=True)
    parser.add_argument("--pretraining_inn_vae",
                        action="store_true",
                        default=False)
    parser.add_argument("--sequential_training",
                        action="store_true",
                        default=False)

    parser.add_argument('--resume_att_classifier',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')
    parser.add_argument('--load_att_classifier',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')

    parser.add_argument("-run_number", type=int, default=1, help="Run number")

    parser.add_argument('--resume_inn_vae',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')
    parser.add_argument('--load_inn_vae',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')

    # Flow params
    parser.add_argument('--num_channels',
                        default=96,
                        type=int,
                        help='Number of channels in Flow++')
    parser.add_argument('--num_ConvAttnBlock',
                        default=1,
                        type=int,
                        help='Number of ConvAttnBlock')
    parser.add_argument('--num_components',
                        default=2,
                        type=int,
                        help='Number of components in the mixture')
    parser.add_argument('--use_self_attn',
                        type=bool,
                        default=True,
                        help='Use attention in the coupling layers')

    # optimization
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min',
                        type=float,
                        default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm',
                        type=float,
                        default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument("--clip",
                        type=float,
                        default=1.0,
                        help="Gradient norm clipping parameter")
    parser.add_argument('--weight_decay_norm_init',
                        type=float,
                        default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal',
                        action='store_true',
                        default=False,
                        help='This flag enables annealing the lambda coefficient from '
                        '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='num of training epochs')
    parser.add_argument("--startepoch",
                        type=int, default=0,
                        help="Sets the first trained epoch for resuming partial training")
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax',
                        action='store_true',
                        default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance',
                        type=str,
                        default='res_mbconv',
                        help='path to the architecture instance')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Dropout probability')

    parser.add_argument('--num_samples',
                        default=64,
                        type=int,
                        help='Number of samples at test time')
    parser.add_argument('--num_workers', default=1,
                        type=int,
                        help='Number of data loader threads')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir',
                        type=str,
                        default='samples',
                        help='Directory for saving samples')

    parser.add_argument('--class_embedding',
                        default='att',
                        type=str)

    args = parser.parse_args()
    args.save = args.result_root + '/eval-' + args.save
    utils.create_exp_dir(args.save)
    args = config_process(args)
    return args


def pretraining_att_classifier(args, model):

    att_trainer = att_classifier_trainer(model)

    common_kwargs = make_training_kwargs(args)
    # logger.info("Starting training MF with specified manifold on NLL")
    learning_curves = att_trainer.train(
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(
            create_filename("checkpoint", None, args))],
        forward_kwargs={},
        custom_kwargs={},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    logger.info("Saving model")
    torch.save(model.state_dict(), create_filename("model", None, args))

    learning_curves = np.vstack(learning_curves).T


def pretraining_inn_vae(args, dataset, model, simulator):
    args = arg_parse()

    dim_v, w2v_att, att, normalize_att = read_attribute()

    attented_classifier = attented_classifier(
        dim_v, w2v_att, att, normalize_att, trainable_w2v=False, non_linear_act=False, normalize_V=False, normalize_F=False)

    common_kwargs = make_training_kwargs(args)

    linear_classifier_trainer = linear_classifier_trainer(attented_classifier)

    # logger.info("Starting training MF with specified manifold on NLL")
    learning_curves = linear_classifier_trainer.train(
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=None,
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(
            create_filename("checkpoint", None, args))],
        forward_kwargs={"mode": "mf"},
        custom_kwargs={"save_output": True},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T


def load_or_resume(args, model, resume=None, load=None):
    model_name = model.__class__.__name__
    create_modelname(args, model_name)

    # Maybe load pretrained model
    if resume:
        resume_filename = create_filename("resume", None, args)
        args.startepoch = args.resume
        logger.info("Resuming training. Loading file %s and continuing with epoch %s.",
                    resume_filename, args.resume + 1)
        model.load_state_dict(
            torch.load(resume_filename, map_location=torch.device("cpu")))
        fix_act_norm_issue(model)
    elif load:
        args_ = copy.deepcopy(args)
        args_.modelname = args.load
        if args_.i > 0:
            args_.modelname += "_run{}".format(args_.i)
        logger.info("Loading model %s and training it on data set %s",
                    args.load, args.modelname, args.dataset)
        model.load_state_dict(
            torch.load(create_filename("model", None, args_), map_location=torch.device("cpu")))
        fix_act_norm_issue(model)
    else:
        logger.info("Training from scratch the model %s on data set %s",
                    model_name, args.dataset)

    return model


def sequential_training(args, dataset, model, simulator):
    args = arg_parse()

    dim_v, w2v_att, att, normalize_att = read_attribute()

    attented_classifier = attented_classifier(
        dim_v, w2v_att, att, normalize_att, trainable_w2v=False, non_linear_act=False, normalize_V=False, normalize_F=False)

    common_kwargs = make_training_kwargs(args)

    linear_classifier_trainer = linear_classifier_trainer(attented_classifier)

    # logger.info("Starting training MF with specified manifold on NLL")
    learning_curves = linear_classifier_trainer.train(
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=None,
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(
            create_filename("checkpoint", None, args))],
        forward_kwargs={"mode": "mf"},
        custom_kwargs={"save_output": True},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T


def create_att_classifier(args):
    dim_v, w2v_att, att, normalize_att = read_attribute()

    att_classifier = discriminative_classifier(
        dim_v, w2v_att, att, normalize_att, trainable_w2v=False, non_linear_act=False, normalize_V=False, normalize_F=False)

    att_classifier = load_or_resume(
        args, att_classifier, args.resume_att_classifier, args.load_att_classifier)

    return att_classifier


def create_inn_vae():
    inn_vae = inn_vae(args)
    inn_vae = load_or_resume(
        args, inn_vae, args.resume_inn_vae, args.load_inn_vae)

    return inn_vae


if __name__ == "__main__":
    # Logger
    args = arg_parse()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
                        datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")
    logger.debug("Starting train.py with arguments %s", args)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    if args.pretraining_att_classifier == True:
        attented_classifier = create_att_classifier(args)
        learning_curves = pretraining_att_classifier(args, attented_classifier)

    elif args.pretraining_inn_vae == True:
        inn_vae = create_inn_vae()
        learning_curves = pretraining_inn_vae(args, inn_vae)

    elif args.sequential_training == True:
        att_classifier = create_att_classifier(args)
        inn_vae = create_inn_vae()
        learning_curves = sequential_training(args, att_classifier, inn_vae)

    # Save

    np.save(create_filename("learning_curve", None, args), learning_curves)

    logger.info("All done! Have a nice day!")
