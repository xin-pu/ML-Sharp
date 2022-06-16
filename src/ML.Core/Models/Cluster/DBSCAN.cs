using System;
using Numpy;

namespace ML.Core.Models
{
    public class DBSCAN : Cluster
    {
        /// <summary>
        ///     密度聚类 DBSCAN
        /// </summary>
        /// <param name="k"></param>
        public DBSCAN(int k)
            : base(k)
        {
        }

        public override NDarray Call(NDarray input)
        {
            throw new NotImplementedException();
        }
    }
}