"""Masked behavioral cloning and critic-only warmup before PPO."""
import torch
from torch import nn


def discounted_returns(rewards, gamma):
    result = torch.empty(len(rewards), dtype=torch.float32)
    value = 0.0
    for index in reversed(range(len(rewards))):
        value = float(rewards[index]) + gamma * value
        result[index] = value
    return result


def fit_actors(actors, observations, masks, actions, epochs=15):
    reports = []
    for agent, actor in enumerate(actors):
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        choice = masks[:, agent].all(dim=-1)
        truth = actions[:, agent]
        positive = torch.where(choice & (truth == 1))[0]
        negative = torch.where(choice & (truth == 0))[0]
        eligible = torch.where(choice)[0]
        if not len(eligible):
            eligible = torch.arange(len(observations))
        history = []
        for _ in range(epochs):
            for _ in range(max(1, (len(eligible)+255)//256)):
                # Rare expert switches must participate in every minibatch.
                # Forced-hold masks already implement their action exactly.
                if len(positive) and len(negative):
                    batch = torch.cat([positive[torch.randint(len(positive), (128,))],
                                       negative[torch.randint(len(negative), (128,))]])
                else:
                    batch = eligible[torch.randint(len(eligible), (256,))]
                logits = actor.network(observations[batch, agent]).masked_fill(~masks[batch, agent], -1e9)
                loss = nn.functional.cross_entropy(logits, actions[batch, agent])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                predictions = actor.distribution(observations[:, agent], masks[:, agent]).probs.argmax(dim=-1)
                logits = actor.network(observations[:, agent]).masked_fill(~masks[:, agent], -1e9)
                history.append(float(nn.functional.cross_entropy(logits, actions[:, agent])))
        confusion = [[int(((truth == a) & (predictions == b)).sum()) for b in (0, 1)] for a in (0, 1)]
        reports.append({'agent_index': agent, 'cross_entropy_by_epoch': history,
            'all_decision_accuracy': float((predictions == truth).float().mean()),
            'legal_choice_accuracy': float((predictions[choice] == truth[choice]).float().mean()) if choice.any() else None,
            'expert_switches': int(truth.sum()), 'legal_choice_decisions': int(choice.sum()),
            'switch_recall': confusion[1][1]/int(truth.sum()) if truth.sum() else None,
            'confusion_true_pred': confusion})
        if len(positive) and confusion[1][1] == 0:
            raise RuntimeError('BC did not learn any expert switches; refusing PPO warm-start')
    return reports


def warm_start(actors, critics, environment_factory, episodes, training_seed, gamma, algorithm):
    base = 1_000_000 + training_seed * 1000
    observations, masks, labels = [], [], []
    for episode in range(episodes):
        environment = environment_factory(base + episode)
        observation = environment.reset()
        try:
            done = False
            while not done:
                mask = environment.action_masks()
                action = environment.pedestrian_actuated_actions()
                assert mask.gather(1, action[:, None]).all()
                observations.append(observation)
                masks.append(mask)
                labels.append(action)
                observation, _, done = environment.step(action)
        finally:
            environment.close()
        print(f'bc_episode={episode+1}/{episodes}', flush=True)
    obs, mask, truth = torch.stack(observations), torch.stack(masks), torch.stack(labels)
    fits = fit_actors(actors, obs, mask, truth)
    actor_parameters = [p.detach().clone() for actor in actors for p in actor.parameters()]
    states, targets = [], []
    agreement = [0, 0]
    choice_agreement = [0, 0]
    rollout_confusion = torch.zeros(len(actors), 2, 2, dtype=torch.long)
    for episode in range(5):
        environment = environment_factory(base + 100 + episode)
        observation = environment.reset()
        episode_states, rewards = [], []
        try:
            done = False
            while not done:
                mask = environment.action_masks()
                expert = environment.pedestrian_actuated_actions()
                with torch.no_grad():
                    action = torch.stack([a.distribution(observation[j], mask[j]).probs.argmax() for j, a in enumerate(actors)])
                choice = mask.all(dim=-1)
                agreement[0] += int((action == expert).sum())
                agreement[1] += len(actors)
                choice_agreement[0] += int(((action == expert) & choice).sum())
                choice_agreement[1] += int(choice.sum())
                for j in range(len(actors)):
                    rollout_confusion[j, int(expert[j]), int(action[j])] += 1
                episode_states.append(observation)
                observation, reward, done = environment.step(action)
                rewards.append(reward)
        finally:
            environment.close()
        states.extend(episode_states)
        targets.append(discounted_returns(rewards, gamma))
        print(f'critic_warmup_episode={episode+1}/5', flush=True)
    states, targets = torch.stack(states), torch.cat(targets)
    critic_report = []
    for index, critic in enumerate(critics):
        inputs = states.flatten(start_dim=1) if algorithm == 'mappo' else states[:, index]
        with torch.no_grad():
            before = float(nn.functional.mse_loss(critic(inputs), targets))
            # Initializing the value bias to the empirical mean keeps the
            # supervised fit in the scale of the unnormalized reward returns.
            critic.network[-1].bias.fill_(targets.mean())
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        for _ in range(15):
            for batch in torch.randperm(len(states)).split(256):
                loss = nn.functional.mse_loss(critic(inputs[batch]), targets[batch])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()
        with torch.no_grad():
            after = float(nn.functional.mse_loss(critic(inputs), targets))
        critic_report.append({'critic_index': index, 'mse_before_warmup': before,
                              'mse_after_warmup': after, 'mean_return_bias_initialization': float(targets.mean())})
    assert all(torch.equal(a, p) for a, p in zip(actor_parameters, [p.detach() for actor in actors for p in actor.parameters()]))
    return {'bc_episodes': episodes, 'bc_simulation_seeds': list(range(base, base+episodes)),
        'bc_transitions_per_agent': len(obs), 'bc_learning_rate': 1e-3, 'bc_epochs': 15,
        'bc_sampling': 'balanced-legal-action-minibatches',
        'expert': 'recorded-pedestrian-actuated-request-rule-at-legal-five-second-decisions',
        'actor_fits': fits, 'critic_warmup_episodes': 5,
        'critic_warmup_simulation_seeds': list(range(base+100, base+105)),
        'critic_warmup_epochs': 15, 'critic_warmup_actor_parameters_unchanged': True,
        'critic_fits': critic_report, 'bc_policy_rollout_expert_agreement': agreement[0]/agreement[1],
        'bc_policy_rollout_confusion_by_agent': rollout_confusion.tolist(),
        'bc_policy_rollout_legal_choice_expert_agreement': choice_agreement[0]/choice_agreement[1] if choice_agreement[1] else None}
