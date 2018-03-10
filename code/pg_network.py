import theano ,theano.tensor as  T
import numpy as np
import lasagne
from collections import OrderedDict

def rmsprop_updates(grads,params,stepsize,rho=0.9,epsilon=1e-9):

    updates = []

    for param, grad in zip(params,grads):
        accum = theano.shared(np.zeros(param.get_value(borrow = True).shape,dtype = param.dtype))
        accum_new = rho * accum + (1-rho) * grad ** 2
        updates.append((accum,accum_new))
        updates.append((param,param + stepsize * grad / T.sqrt(accum_new + epsilon)))

    return updates


class PGLearner:

    def __init__(self,pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = T.tensor4('states')  # states is [batch_size,channel_size,input_height,input_width ]
        actions = T.lvector('actions') # actions shape is batch_size.It is action that network choose
        values = T.vector('values')    # values shape is batch_size. It is value that action stand for.

        print('network_input_height=', pa.network_input_height)
        print('network_input_width=', pa.network_input_width)
        print('network_output_dim=', pa.network_output_dim)

        # image represent
        self.l_out = build_pg_network(self.input_height,self.input_width,self.output_height)

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        params = lasagne.layers.helper.get_all_params(self.l_out)

        print('params=',params,'counts',lasagne.layers.count_params(self.l_out))

        self._get_param = theano.function([],params)

        # =====================================
        #    Training
        #======================================

        prob_act = lasagne.layers.get_output(self.l_out,states) # shape is [batch_size ,output_height].It is action prob.

        self._get_act_prob = theano.function([states],prob_act,allow_input_downcast= True)

        #=======================================
        # policy gradients
        #=======================================

        N = states.shape[0]

        loss = T.log(prob_act[T.arange(N),actions]).dot(values) / N

        grads = T.grad(loss,params)

        updates = rmsprop_updates(grads,params,self.lr_rate,self.rms_rho,self.rms_eps)

        self._train_fn = theano.function([states,actions,values],loss,updates = updates,allow_input_downcast=
                                         True)

        self._get_loss = theano.function([states,actions,values],loss,
                                         allow_input_downcast= True)
        self._get_grad = theano.function([states,actions,values],grads)

        # -------supervised learning --------------------

        su_target = T.ivector('su_target')
        su_loss = lasagne.objectives.categorical_crossentropy(prob_act,su_target)
        su_loss = su_loss.mean()

        l2_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l2)
        # l1_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l1)

        su_loss += 1e-3 * l2_penalty
        print('lr_rate=', self.lr_rate)

        su_updates = lasagne.updates.rmsprop(su_loss, params,
                                             self.lr_rate, self.rms_rho, self.rms_eps)
        self._su_train_fn = theano.function([states, su_target], [su_loss, prob_act], updates=su_updates)

        self._su_loss = theano.function([states, su_target], [su_loss, prob_act])

        self._debug = theano.function([states], [states.flatten(2)])
    def choose_action(self,state):

        act_prob = self.get_one_act_prob(state)

        csprob_n = np.cumsum(act_prob)
        act = (csprob_n > np.random.rand()).argmax()

        return act

    def train(self,states,actions,values):

        loss = self._train_fn(states,actions,values)
        return loss

    def get_params(self):

        return self._get_param()

    def get_grad(self,states,actions,values):

        return self._get_grad(states,actions,values)

    def get_one_act_prob(self,state):

        states = np.zeros((1,1,self.input_height,int(self.input_width)),dtype=theano.config.floatX)
        states[0,:,:] = state
        act_prob = self._get_act_prob(states)[0]

        return act_prob

    def get_act_probs(self,states):

        act_probs = self._get_act_prob(states)

        return act_probs

    #====================================
    # supervised learning
    #====================================
    def su_train(self,states,target):
        loss, prob_act = self._su_train_fn(states, target)
        return np.sqrt(loss), prob_act

    def su_test(self,states,target):
        loss, prob_act = self._su_loss(states, target)
        return np.sqrt(loss), prob_act


    #====================================
    #  save and load network
    #====================================

    def return_net_params(self):
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_net_params(self,net_params):
        lasagne.layers.helper.set_all_param_values(self.l_out,net_params)


def build_pg_network(input_height,input_width,output_dim):

    l_in = lasagne.layers.InputLayer(
        shape=(None,1,input_height,input_width)
    )

    l_hid = lasagne.layers.DenseLayer(
        l_in,
        num_units = 20,
        nonlinearity=lasagne.nonlinearities.rectify,
        W= lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hid,
        num_units = output_dim,
        nonlinearity= lasagne.nonlinearities.softmax,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )
    return l_out

