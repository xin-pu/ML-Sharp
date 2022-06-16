using System;
using Numpy;

namespace ML.Core.Models
{
    public class Spectral : Cluster
    {
        /// <summary>
        ///     谱聚类
        /// </summary>
        /// <param name="k"></param>
        public Spectral(int k) : base(k)
        {
        }

        public override NDarray Call(NDarray input)
        {
            throw new NotImplementedException();
        }
    }
}