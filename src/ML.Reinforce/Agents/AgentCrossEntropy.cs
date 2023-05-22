using ML.Core.Models;
using ML.Reinforce.Models;
using Numpy;

namespace ML.Reinforce.Agents
{
    public class AgentCrossEntropy : Agent
    {
        public override IModel LearnPolicy(Reward[] rewards)
        {
            throw new NotImplementedException();
        }

        public override NDarray RunPolicy(State state)
        {
            throw new NotImplementedException();
        }
    }
}