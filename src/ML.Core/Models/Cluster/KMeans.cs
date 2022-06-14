using System;
using Numpy;

namespace ML.Core.Models
{
    public class KMeans : Cluster
    {
        public KMeans(int k) : base(k)
        {
        }

        public override NDarray Call(NDarray input)
        {
            throw new NotImplementedException();
        }
    }
}