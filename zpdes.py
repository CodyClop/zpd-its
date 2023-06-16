import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

""" 
    Corresponds to v_i,j
    Represents a possible value for a given parameter
    * label : value name 
    * code : identifier
    * weight : weight associated to the value for the MAB
    * next : name of the following group to be instanciated
    * active : whether this value is currently activated (can be drawn from MAB)
    * activation : step of ZPD expansion at which the value gets activated
"""
class Value: 
    def __init__(self, label, code, weight=0, next=None, active=True, activation=1):
        self.label = label
        self.code = code
        self.next = next
        self.active = active
        self.weight = weight
        self.scores = [] # history of passed scores when this value was involved
        self.activation = activation 
    
    def success_rate(self):
        if len(self.scores) == 0:
            return 0
        if len(self.scores) < 4:
            return np.mean(self.scores)
        return np.mean(self.scores[-4:])
    
    def set_param(self, param):
        self.param = param

    def set_group(self, group):
        self.group = group

    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self):
        return '(' + self.label + ', weight: ' + str(self.weight) + ', success_rate: ' + str(self.success_rate()) + (' ' if self.active else ' in') + 'active' + ', next: ' + str(self.next) + ')'


""" 
    Corresponds to a_i 
    Represents a parameter in a group
    * label : parameter name
    * group : group the parameter belongs to
"""
class Param:
    def __init__(self, label, group):
        self.label = label
        self.values = [] # possible values of this parameter
        self.group = group

    def add_value(self, value):
        value.set_param(self)
        value.set_group(self.group)
        self.values.append(value)

    def weight(self):
        return [value.weight for value in self.values]
    
    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self):
        return '(Param ' + self.label + ': \n\t\t' + '\n\t\t'.join(map(str,self.values)) + ')'
        
    
""" 
    Corresponds to H_x 
    Represents a group of parameters
    * label : group name
"""
class Group:
    def __init__(self, label):
        self.label = label
        self.params_dict = {} # keys = param names, values = param objects

    def params(self):
        return list(self.params_dict.values())
    
    def weight(self):
        return [param.weight() for param in self.params_dict.values()]
    
    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self):
        return '(Group ' + self.label + ': \n\t' + '\n\t'.join(map(str,self.params_dict.values())) + ')'
    

class ActivitySpace:
    def __init__(self):
        self.zpd_timestamp = 1 # step in the expansion of zpd
        self.groups_dict = {}  # keys = group names, values = group objects

    def groups(self):
        return list(self.groups_dict.values())
    
    # Weights for MABs
    def weights(self):
        return [group.weight() for group in self.groups()]

    def add_values(self, group_label, param_label, values):
        if group_label not in self.groups_dict.keys():
            self.groups_dict[group_label] = Group(group_label)

        group = self.groups_dict[group_label]

        if param_label not in group.params_dict.keys():
            group.params_dict[param_label] = Param(param_label, group)

        param = group.params_dict[param_label]

        for value in values:
            param.add_value(value)
            
    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self):
        return '\n'.join(map(str, self.groups_dict.values()))
    

"""
    * gamma : exploration rate 
    * d : number of samples used to compute reward
    * lambdaZPD : success rate needed to expand the ZPD
    * lambdaA : success rate needed to deactivate a value
    * beta : applied to updated arm's weight
    * eta : applied to the reward to update arm
"""
class ZPDES:
    def __init__(self, gamma, d, lambdaZPD, lambdaA, beta, eta):
        self.init_activity_space()
        self.gamma = gamma
        self.d = d
        self.lambdaZPD = lambdaZPD
        self.lambdaA = lambdaA
        self.beta = beta
        self.eta = eta

    def init_activity_space(self):
        self.A_S = ActivitySpace()
        self.A_S.add_values('Type', 'Type', [Value('souplesse', 'S', next='Souplesse Mobilité'), 
                                             Value('qualité', 'Q', active=False, activation=2, next='Qualité Mouvement Niveau')])

        self.A_S.add_values('Souplesse Mobilité', 'Mobilité', [Value('extension', 'E', next='Souplesse Extension'), 
                                                               Value('latéro-flexion', 'LF', next='Souplesse Latéro-Flexion'),
                                                               Value('rotation', 'R', next='Souplesse Niveau')])

        self.A_S.add_values('Souplesse Extension', 'Mouvement', [Value('cache-tête', 'CT', next='Souplesse Niveau'), 
                                                                 Value('arc-arrière', 'AA', next='Souplesse Niveau')])

        self.A_S.add_values('Souplesse Latéro-Flexion', 'Position', [Value('assis', 'A', next='Souplesse Niveau'), 
                                                                     Value('debout', 'D', next='Souplesse Niveau')])

        self.A_S.add_values('Souplesse Niveau', 'Niveau', [Value('Niveau 1', '1'), 
                                                           Value('Niveau 2', '2', active=False, activation=3), 
                                                           Value('Niveau 3', '3', active=False, activation=5)])

        self.A_S.add_values('Qualité Mouvement Niveau', 'Niveau', [Value('Mouvement Niveau 1', 'A', activation=2, active=False, next='Qualité Mouvement 1'),
                                                                   Value('Mouvement Niveau 2', '', activation=3, active=False, next='Qualité Mouvement 2'),
                                                                   Value('Mouvement Niveau 3', 'PLR', activation=5, active=False, next='Qualité Mouvement 3')])

        self.A_S.add_values('Qualité Mouvement 1', 'Niveau', [Value('Niveau 1', '1', active=False, activation=2), 
                                                              Value('Niveau 2', '2', active=False, activation=3), 
                                                              Value('Niveau 3', '3', active=False, activation=4)])

        self.A_S.add_values('Qualité Mouvement 2', 'Mouvement', [Value('eq. assis rotations', 'R', active=False, activation=3), 
                                                                 Value('eq. assis pied levé', 'PL', active=False, activation=3),
                                                                 Value('eq. assis mvmnts latéraux', 'ML', active=False, activation=3),
                                                                 Value('eq. assis pivot', 'P', active=False, activation=3)])

        self.A_S.add_values('Qualité Mouvement 2', 'Niveau', [Value('niveau 1', '1', active=False, activation=3), 
                                                              Value('niveau 2', '2', active=False, activation=5),
                                                              Value('niveau 3', '3', active=False, activation=6)])

        self.A_S.add_values('Qualité Mouvement 3', 'Niveau', [Value('niveau 1', '1', active=False, activation=5), 
                                                              Value('niveau 2', '2', active=False, activation=6),
                                                              Value('niveau 3', '3', active=False, activation=7)])
        
    def sampleValues(self, Hx, Wx):
        h_x = []
        for i in range(len(Hx)):
            candidates = [] # activated values
            weights = [] # activated values' weights
            for j in range(len(Wx[i])):
                if Hx[i].values[j].active:
                    candidates.append(Hx[i].values[j])
                    weights.append(Wx[i][j])
            w_i = softmax(weights)
            xi_u = np.ones(len(weights), dtype='float64')
            xi_u /= np.sum(xi_u) # Uniform distribution over active values
            p_i = w_i*(1-self.gamma) + self.gamma * xi_u
            u_i = np.random.choice(candidates, p=p_i)
            h_x.append(u_i)
        return h_x
    
    def genActivity(self):
        H, W = self.A_S.groups(), self.A_S.weights()
        e = []
        h_1 = self.sampleValues(H[0].params(), W[0])
        next_group = h_1[-1].next
        e.append(h_1)
        while next_group != None:
            h_i = self.sampleValues(self.A_S.groups_dict[next_group].params(), self.A_S.groups_dict[next_group].weight())
            next_group = h_i[-1].next
            e.append(h_i)
        return e, ''.join([value.code for group in e for value in group])
    

    def computeReward(self, e, C):
        r = []
        for h_x in e:
            for value in h_x:
                value.scores.append(C)
                d_ = min(len(value.scores), self.d)
                if d_ > 1:
                    r_x = np.sum(value.scores[-int(np.ceil(d_/2)):]) / np.ceil(d_/2) # Average scores of d_/2 last exercices for this group
                    r_x -= np.sum(value.scores[-d_:-int(np.ceil(d_/2))]) / (np.floor(d_/2) if d_ !=1 else 1) # Average scores of indexes -d_ -> -d_/2 
                else:
                    r_x = 0
                r.append(r_x)
        return r
    

    def updateZPD(self):
        # success_rates = []
        expand = True
        for group in self.A_S.groups():
            for param in group.params():
                for value in param.values:
                    if value.active and value.success_rate() < self.lambdaZPD:
                        expand = False
                        
        if expand:
            #expand zpd
            self.A_S.zpd_timestamp += 1

            # activate values
            for group in self.A_S.groups():
                for param in group.params():
                    for value in param.values:
                        if self.A_S.zpd_timestamp == value.activation:
                            value.active = True
                            # value's weight is initialized at the minimum active weight of parameter
                            value.weight = np.min([value.weight if value.active else 1000 for value in param.values])

        # Deactivation rules
        for value in self.A_S.groups_dict['Souplesse Niveau'].params_dict['Niveau'].values:
            if value.label == 'Niveau 1' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 3:
                value.active = False 

            if value.label == 'Niveau 2' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 5:
                value.active = False 
        
        for value in self.A_S.groups_dict['Qualité Mouvement Niveau'].params_dict['Niveau'].values:
            if value.label == 'Mouvement Niveau 1' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 5:
                value.active = False 

        for value in self.A_S.groups_dict['Qualité Mouvement 1'].params_dict['Niveau'].values:
            if value.label == 'Niveau 1' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 3:
                value.active = False

            if value.label == 'Niveau 2' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 4:
                value.active = False

        for value in self.A_S.groups_dict['Qualité Mouvement 2'].params_dict['Niveau'].values:
            if value.label == 'Niveau 1' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 5:
                value.active = False

            if value.label == 'Niveau 2' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 6:
                value.active = False

        for value in self.A_S.groups_dict['Qualité Mouvement 3'].params_dict['Niveau'].values:
            if value.label == 'Niveau 1' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 6:
                value.active = False

            if value.label == 'Niveau 2' and value.success_rate() > self.lambdaA and self.A_S.zpd_timestamp >= 7:
                value.active = False


    def update(self, e, C):
        r = self.computeReward(e, C)
        
        # Update greedy experts
        i = 0
        for h_x in e:
            for u_i in h_x:
                u_i.weight = self.beta * u_i.weight + self.eta * r[i]
                i += 1

        self.updateZPD()
    