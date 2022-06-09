using System;
using System.Linq;
using System.Reflection;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Optimizers;

namespace ML.Guide.DataProvider
{
    public class DataProvider
    {
        /// <summary>
        ///     Get All Metric Type
        /// </summary>
        /// <returns></returns>
        public Type[] GetMerticTypes()
        {
            return GetSubType(typeof(Metric));
        }

        public Type[] GetOptimizerTypes()
        {
            return GetSubType(typeof(Optimizer));
        }

        public Type[] GetLossTypes()
        {
            return GetSubType(typeof(Loss));
        }

        internal Type[] GetSubType(Type type)
        {
            var assembly = Assembly.GetAssembly(type);
            var baseType = assembly?.ExportedTypes
                .Where(t => t.IsSubclassOf(type) && !t.IsAbstract && t.IsPublic)
                .OrderBy(t => t.Name)
                .ToArray();
            return baseType;
        }
    }
}