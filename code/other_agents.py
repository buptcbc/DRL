import numpy as np

def get_sjf_action(machine,job_slot,pa):
    sjf_score = 0
    act = len(job_slot.slot)

    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]

        if new_job is not None:

            avbl_mec_slot = machine.avl_mec_slot[new_job.job_net_len:new_job.job_net_len + new_job.job_mec_len]
            avbl_net_slot = machine.avl_net_slot[:new_job.job_net_len]
            mec_res_left = avbl_mec_slot - new_job.res
            mec_net_left = avbl_net_slot - pa.res_slot

            if np.all(mec_res_left[:]>=0) and np.all(mec_net_left[:] >=0 ): # there is enough resource

                tmp_sjf_score = 1 / float(new_job.job_mec_len)

                if tmp_sjf_score > sjf_score :
                    sjf_score = tmp_sjf_score
                    act = i
    return act

def get_random_action(job_slot):
    num_act = len(job_slot.slot) + 1
    act = np.random.randint(num_act)

    return act
