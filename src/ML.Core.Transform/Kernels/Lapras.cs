using System;
using System.Linq;
using ML.Core.Transform;
using Numpy;

namespace MLNet.Transforms
{
    /// <summary>
    ///     h(i,j) =  e^(-beta*Norm1(xi-xj))
    /// </summary>
    /// <param name="beta"></param>
    /// <exception cref="ArgumentException"></exception>
    public class Lapras : Kernel
    {
        public Lapras(double beta = 1)
            : base(KernelType.Lapras)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

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
                    var delta = x_array[i] - input;
                    var res = np.linalg.norm(delta, 1, -1);
                    output[i] = np.exp(-Beta * res);
                });

            return output;
        }
    }
}