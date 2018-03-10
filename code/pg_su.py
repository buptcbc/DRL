import numpy as np
import theano
import time
import sys
import _pickle as  cPickle

import environment
import pg_network
import other_agents
import job

np.set_printoptions(threshold='nan')


def add_sample(X, y, idx, X_to_add, y_to_add):
    X[idx, 0, :, :] = X_to_add
    y[idx] = y_to_add


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def launch(pa, pg_resume=None, render=False, end='no_new_job'):

    env = environment.environment(pa, render=False, end=end)

    pg_learner = pg_network.PGLearner(pa)

    if pg_resume is not None:
        net_handle = open(pg_resume, 'r')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    if pa.evaluate_policy_name == "SJF":
        evaluate_policy = other_agents.get_sjf_action
    else:
        print("Panic: no policy known to evaluate.")
        exit(1)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------

    nw_mec_len_seqs, nw_net_len_seqs, nw_seq_size = job.generate_work_sequence(pa,seed=24)

    # print 'nw_time_seqs=', nw_len_seqs
    # print 'nw_size_seqs=', nw_size_seqs

    mem_alloc = 4

    X = np.zeros([pa.simu_len * pa.num_ex * mem_alloc * pa.num_user, 1,
                  pa.network_input_height, int(pa.network_input_width)],
                 dtype=theano.config.floatX)
    y = np.zeros(pa.simu_len * pa.num_ex * mem_alloc * pa.num_user,
                 dtype='int32')

    print ('network_input_height=', pa.network_input_height)
    print ('network_input_width=', pa.network_input_width)

    counter = 0

    for train_ex in range(pa.num_ex):

        env.reset()

        for _ in range(pa.episode_max_length):

            # ---- get current state ----
            ob = env.observe()

            a = evaluate_policy(env.machine, env.job_slot,pa)

            if counter < pa.simu_len * pa.num_ex * mem_alloc * pa.num_user:

                add_sample(X, y, counter, ob, a)
                counter += 1

            ob, rew, done, info = env.step(a)

            if done:  # hit void action, exit
                break

        # roll to next example
        env.seq_ex = (env.seq_ex + 1) % env.pa.num_ex
    print('counter',counter)

    num_train = int(0.8 * counter)
    num_test = int(0.2 * counter)

    X_train, X_test = X[:num_train], X[num_train: num_train + num_test]
    y_train, y_test = y[:num_train], y[num_train: num_train + num_test]

    # Normalization, make sure nothing becomes NaN

    # X_mean = np.average(X[:num_train + num_test], axis=0)
    # X_std = np.std(X[:num_train + num_test], axis=0)
    #
    # X_train = (X_train - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std

    # ----------------------------
    print("Start training...")
    # ----------------------------

    for epoch in range(pa.num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, pa.batch_size, shuffle=True):
            inputs, targets = batch
            # print(inputs.shape, targets.shape)
            err, prob_act = pg_learner.su_train(inputs, targets)
            pg_act = np.argmax(prob_act, axis=1)
            train_err += err
            train_acc += np.sum(pg_act == targets)
            train_batches += 1

        # # And a full pass over the test data:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, pa.batch_size, shuffle=False):
            inputs, targets = batch
            err, prob_act = pg_learner.su_test(inputs, targets)
            pg_act = np.argmax(prob_act, axis=1)
            test_err += err
            test_acc += np.sum(pg_act == targets)
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, pa.num_epochs, time.time() - start_time))
        print("  training loss:    \t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / float(num_train) * 100))
        print("  test loss:        \t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:    \t\t{:.2f} %".format(
            test_acc / float(num_test) * 100))

        sys.stdout.flush()

        if epoch % pa.output_freq == 0:

            net_file = open(pa.output_filename + '_net_file_' + str(epoch) + '.pkl', 'wb')
            cPickle.dump(pg_learner.return_net_params(), net_file, -1)
            net_file.close()

    print("done")

def main():
    import parameters
    pa = parameters.parameters()

    pa.simu_len = 50
    pa.num_ex = 100  # 100
    pa.num_seq_per_batch = 10
    pa.output_freq = 10
    pa.num_epochs=2000
    pa.new_job_rate = 0.1
    pg_resume = None

    render = False

    launch(pa, pg_resume, render, end='all_done')

if __name__ == '__main__':
    main()