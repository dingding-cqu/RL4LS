import random

from core.problem import Problem

def creat_select_cc():
    nb_instances = 20
    pth = '../problems4cc/valid'
    for i in range(nb_instances):
        nb_agents = random.randint(20, 30)
        p1 = random.random() * 0.5 + 0.1
        p = Problem()
        dom_max, dom_min = 8, 2
        p.random_domain_binary(nb_agents, [ele for ele in range(dom_min,dom_max+1,1)], p1)
        p.save(f'{pth}/{i}.xml')


def creat_MS():
    nb_instances = 20
    pth = '../problems/MS'
    # for i in range(nb_instances):
    p = Problem()
    p.random_meeting_scheduling(nb_people= 90, nb_meeting=20, nb_slots=20,
                                nb_select_meetings=2, min_travel_time=6, max_travel_time=10)
    # p.save(f'{pth}/{i}.xml')

def creat_trainproblems():
    nb_instances = 200
    pth = '../problems/train'
    for i in range(nb_instances):
        nb_agents = random.randint(40, 60)
        p1 = random.random() * 0.25
        p1 = max(p1, .1)
        dom_size = random.randint(3, 15)
        p = Problem()
        p.random_binary(nb_agents, dom_size, p1)
        p.save(f'{pth}/{i}.xml')

if __name__ == '__main__':
    creat_select_cc()