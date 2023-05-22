using CommunityToolkit.Mvvm.ComponentModel;
using ML.Core.Models;
using Numpy;

namespace ML.Reinforce.Models
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent : ObservableObject
    {
        private IModel? _policy;
        private Reward[] _rewards = Array.Empty<Reward>();

        protected Agent()
        {
            _policy = null;
        }

        public Reward[] Rewards
        {
            set => SetProperty(ref _rewards, value);
            get => _rewards;
        }

        public IModel? Policy
        {
            set => SetProperty(ref _policy, value);
            get => _policy;
        }

        /// <summary>
        ///     根据rewards 学习或者更新策略
        /// </summary>
        public abstract IModel LearnPolicy(Reward[] rewards);


        /// <summary>
        ///     策略函数，根据最新状态，生成新的动作
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public abstract NDarray RunPolicy(State state);
    }
}