import os

from learner.act_r_custom import ActRMeaning

from teacher.avya import AvyaTeacher

import multiprocessing as mp

import simulation.run

import plot.simulation

from utils.utils import dic2string, dump, load

import argparse


def main(student_model=None, teacher_model=None,
         student_param=None,
         n_item=30, grades=(1, ), t_max=2000,
         max_iter=10, n_cpu=mp.cpu_count()-1,
         normalize_similarity=True, force=False, plot_fig=True,
         init_eval=20, verbose=True
         ):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = AvyaTeacher

    extension = \
        f'full_stack_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
        f'norm_{normalize_similarity}_' \
        f'max_iter_{max_iter}'

    bkp_file = os.path.join('bkp', 'full_stack', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = simulation.run.with_bayesian_opt(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item,
            grades=grades,
            t_max=t_max,
            normalize_similarity=normalize_similarity,
            max_iter=max_iter,
            n_cpu=n_cpu,
            init_eval=init_eval,
            verbose=verbose
        )

        dump(r, bkp_file)

    if plot_fig:
        plot.simulation.summary(
            p_recall=r['p_recall'],
            seen=r['seen'],
            successes=r['successes'],
            extension=extension)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_fig', '-n', action='store_true', default=False,
                        dest='no_fig',
                        help='Do not create fig')

    parser.add_argument('--n_cpu', '-c', default=mp.cpu_count(),
                        dest='n_cpu',
                        help='Number of cpu to use', type=int)

    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        dest='verbose',
                        help='Verbose')

    args = parser.parse_args()
    main(plot_fig=not args.no_fig, n_cpu=args.n_cpu, verbose=args.verbose)
