using System;
using System.Linq;
using Numpy;

namespace ML.Core.Transform
{
    /// <summary>
    ///     h(i,j) =  tanh ( beta *dot( xi,xj) + theta)
    /// </summary>
    /// <param name="beta"></param>
    /// <exception cref="ArgumentException"></exception>
    public class Sigmoid : Kernel
    {
        public Sigmoid(double beta = 0.5, double theta = 0.5)
            : base(KernelType.Sigmoid)
        {
            if (beta <= 0.0 || theta >= 0)
                throw new Exception("Please give Beta > 0 and Theta < 0");
            Beta = beta;
            Theta = theta;
        }

        public double Beta { protected set; get; }

        public double Theta { protected set; get; }

        public override NDarray Call(NDarray input)
        {
            var p = input.shape[0];

            var output = np.zeros(p, p);


            var x_array = Enumerable.Range(0, p)
                .Select(r => input[$"{r},:"]).ToList();

            Enumerable.Range(0, p)
                .AsParallel()
                .ToList()
                .ForEach(i =>
                {
                    var res = np.dot(input, x_array[i]);
                    output[i] = np.tanh(Beta * res + Theta);
                });

            return output;
        }
    }
}