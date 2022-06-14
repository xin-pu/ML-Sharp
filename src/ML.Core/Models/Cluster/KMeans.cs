using System.Linq;
using Numpy;
using Numpy.Models;

namespace ML.Core.Models
{
    public class KMeans : Cluster
    {
        public KMeans(int k) : base(k)
        {
        }

        public override NDarray Call(NDarray input)
        {
            var batchsize = input.shape[0];
            var temp_centroid_group = np.zeros(new Shape(batchsize, 1), np.int64);
            var kmeans = np.array(Enumerable.Range(0, K).Select(i => np.random.choice(batchsize)));
            return temp_centroid_group;
        }
    }
}