using System;
using ML.Core.Metrics.Regression;

namespace ML.Guide.DataProvider
{
    public class DataProvider
    {
        /// <summary>
        ///     Todo Get Type[] by reflect
        /// </summary>
        /// <returns></returns>
        public Type[] GetMerticTypes()
        {
            return new[]
            {
                typeof(MeanAbsolutePercentageError),
                typeof(MeanAbsoluteError),
                typeof(MeanSquaredError)
            };
        }
    }
}