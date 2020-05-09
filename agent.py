#Using Deep Learning

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent():
    
    def __init__ (self, state_size, is_eval=False, model_name=""):
        self.__inventory = []  # Includes all stocks currently
        self.__total_profit = 0
        self.action_history = []  # all action taken
        self.state_size = state_size # normalized previous days
        
        self.action_size = 3 # stall, buy, sell
        self.memory = deque(maxlen = 1000)
        self.model_name = model_name
        self.is_eval = is_eval
        
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.model = load_model("models/"+model_name) if is_eval else self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(units = 32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(units = 8, activation = 'relu'))
        model.add(Dense(units = self.action_size, activation='linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
        return model
    
    def reset(self):
        
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        
    # action to be taken any state
    def act(self,state, price_data):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            
        else:
            options = self.model.predict(state) # predict q-value of the current state
            
            # pick the action with highest probability
            action = np.argmax(options[0])# select the q-value with highest value
            
            
            
        bought_price = None # Price of at which stock was bought
        if action == 0: # Do Nothing
            print(".", end = ' ', flush = True)
            self.action_history.append(action)
            
        elif action == 1: # Buy
            
            print(price_data)
            self.buy(price_data)
            self.action_history.append(action)
            
        elif action == 2 and self.has_inventory(): # Sell
            bought_price = self.sell(price_data)
            self.action_history.append(action)
            
        else: # action is 2 (sell) but we don't have anything in inventory to sell!
            self.action_history.append(0)
            
        return action, bought_price
        
    def buy(self, price_data):
        self.__inventory.append(price_data)
        print("Buy : {}".format(self.format_price(price_data)))
        
    def sell(self, price_data):
        bought_price = self.__inventory.pop(0) # Selling 1st item
        profit = price_data - bought_price
        self.__total_profit += profit
        print("Sell : {} | Profit : {}".format(self.format_price(price_data), self.format_price(bought_price)))
        return bought_price
    
    
    def has_inventory(self):
        return (len(self.__inventory) > 0)
    
    def format_price(self, n):
        return ("-$" if n<0 else "+$") + "{0:.2f}".format(abs(n))
    
    def get_total_profit(self):
        return self.format_price(self.__total_profit)
    
    def experience_replay(self, batch_size):
        mini_batch=[]
        l = len(self.memory)
        for i in range(l-batch_size + 1,l):
            mini_batch.append(self.memory[i])
            
        for state, action, reward, next_state, done in mini_batch:
            if done :
                target = reward
                
            else :
                # updated q_value = reward + gamma * [max_a' Q(s',a')]
                next_q_values = self.model.predict(next_state)[0] # this is Q(s', a') for all possible a'
                #  update target q_value using Bellman equation
                target= reward + self.gamma* np.amax(next_q_values) # max value of all 
            
            predicted_target = self.model.predict(state) # predict q_value for current state
            predicted_target[0][action] = target # Substitue target q_value to the predicted value
            # Train the model with updated action values
            self.model.fit(state, predicted_target,epochs = 1, verbose = 0) # train the model with new q_value
            
        # Makes epsilon smaller over time, so do more exploitation than exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
    
        
        
       
    

        