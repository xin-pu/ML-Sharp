using Numpy;

namespace ML.Utilty
{
    public class nn
    {
        public static NDarray sigmoid(NDarray input)
        {
            return 1.0 / (1 + np.exp(-input));
        }


        public static NDarray tanh(NDarray input)
        {
            return 2.0 / (1 + np.exp(-input)) - 1;
        }
    }
}