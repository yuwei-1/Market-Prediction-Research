import unittest
import sys
import gymnasium as gym
import os
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from models.deep_q_agent import DQNAgent
import torch


class BaseDQNTests:
    
    class TestDQN(unittest.TestCase):

        def test_double_dqn_init(self):
            dqn = DQNAgent(1,1)
            self.assertNotIn('target_net', dqn.__dict__)
            dqn = DQNAgent(1,1,double_dqn=True)
            self.assertIn('target_net', dqn.__dict__)

        def test_save_functionality(self):
            test_file_name = "test.pt"
            self.dqn.save(test_file_name)
            self.assertTrue(os.path.exists(test_file_name))
            if os.path.exists(test_file_name):
                os.remove(test_file_name)
        
        def test_load_functionality(self):
            self.dqn.load(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal\models\tests\test_model.pt")
            for param in self.dqn.net.parameters():
                self.assertEqual(param.mean().item(), 0)
            for param in self.dqn.target_net.parameters():
                self.assertEqual(param.mean().item(), 0)

        def test_hard_update(self):
            dqn_state_dict = self.dqn.net.state_dict()
            for key in dqn_state_dict.keys():
                dqn_state_dict[key] = torch.zeros_like(dqn_state_dict[key])
            self.dqn.net.load_state_dict(dqn_state_dict)
            steps_done = 500
            self.dqn.perform_hard_target_update(steps_done)
            target_dict = self.dqn.target_net.state_dict()
            for key in target_dict.keys():
                self.assertEqual(target_dict[key].sum(), 0)

        def test_soft_update(self):
            dqn_state_dict = self.dqn.net.state_dict()
            target_dict = self.dqn.target_net.state_dict()
            for key in dqn_state_dict.keys():
                dqn_state_dict[key] = torch.zeros_like(dqn_state_dict[key])
                target_dict[key] = torch.ones_like(target_dict[key])
            
            self.dqn.target_net.load_state_dict(target_dict)
            self.dqn.net.load_state_dict(dqn_state_dict)

            self.dqn.tau = 0.1
            self.dqn.perform_soft_target_update()
            
            target_dict = self.dqn.target_net.state_dict()
            for key in target_dict.keys():
                self.assertTrue((target_dict[key] == 1-self.dqn.tau).all())

        # def test_qagent_training(self):
        #     #self.dqn = 
        #     #DQNAgent(1, 1, double_dqn=True, environment=gym.make('CartPole-v1', render_mode="none"))
        #     eps = 1
        #     discount = 0.99
        #     title="testing train method"
        #     batch_size=self.batch_size
        #     train_threshold=self.train_threshold
        #     learning_rate=1
        #     optimizer=torch.optim.Adam
        #     loss=torch.nn.MSELoss()

        #     self.assertEqual(len(self.dqn.replay_memory), 0)

        #     self.dqn.train()
        #         # episodes=eps, 
        #         # discount=discount,
        #         # title=title,
        #         # batch_size=batch_size, 
        #         # train_threshold=train_threshold, 
        #         # learning_rate=learning_rate,
        #         # optimizer=optimizer, 
        #         # loss=loss,
        #         # plot=False)
            
        #     self.assertNotEqual(self.dqn.get_memory_len(), 0)
        #     self.assertEqual(len(self.dqn.episode_durations), eps)

        #     first_mem = self.dqn.replay_memory[0]

        #     self.assertIsInstance(first_mem, tuple)
        #     self.assertIsInstance(first_mem.state, torch.Tensor)
        #     self.assertIsInstance(first_mem.action, torch.Tensor)
        #     self.assertIsInstance(first_mem.next_state, torch.Tensor)
        #     self.assertIsInstance(first_mem.reward, torch.Tensor)

        #     self.assertEqual(first_mem.state.dtype, torch.float32)
        #     self.assertEqual(first_mem.next_state.dtype, torch.float32)

        #     self.assertEqual(first_mem.state.device.type, 
        #                     first_mem.action.device.type)
        #     self.assertEqual(first_mem.state.device.type,
        #                     first_mem.next_state.device.type)
        #     self.assertEqual(first_mem.state.device.type,
        #                     first_mem.reward.device.type)
        #     self.assertEqual(first_mem.state.device.type,
        #                     self.dqn.device.type)
            

        def test_retrieve_batch_info(self):
            experience_count = 100
            state_dim=2
            reward_dim = 1
            action_dim = 1
            batch_size = 50

            states_shape = (batch_size, state_dim)
            reward_shape = (batch_size, reward_dim)
            action_shape = (batch_size, action_dim)

            dummy_replay_mem = self.dqn.init_agent_memory()
            self.dqn.replay_memory = dummy_replay_mem

            # create some dummy experience data
            state = torch.zeros(2)
            next_state = torch.zeros(2)
            reward = torch.zeros(1)
            action = torch.zeros(1)

            for i in range(experience_count):
                self.dqn.retain_experience(state, action, next_state, reward)
            
            self.dqn.batch_size = batch_size
            states, actions, next_states, rewards = self.dqn.retrieve_batch_info()

            self.assertIsInstance(states, torch.Tensor)
            self.assertIsInstance(actions, torch.Tensor)
            self.assertIsInstance(rewards, torch.Tensor)
            self.assertIsInstance(next_states, tuple)
            self.assertIsInstance(next_states[0], torch.Tensor)

            self.assertEqual(states.shape, states_shape)
            self.assertEqual(len(next_states), batch_size)
            self.assertEqual(rewards.shape, reward_shape)
            self.assertEqual(actions.shape, action_shape)


if __name__ == "__main__":
    unittest.main()