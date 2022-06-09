using System;
using System.Linq;
using System.Reflection;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;

namespace ML.Guide.DataProvider
{
    public class DataProvider
    {
        /// <summary>
        ///     Get All GDModels
        /// </summary>
        /// <returns></returns>
        public Type[] GetGDModelTypes()
        {
            return GetSubType(typeof(ModelGD));
        }

        /// <summary>
        ///     Get All Optimizer
        /// </summary>
        /// <returns></returns>
        public Type[] GetOptimizerTypes()
        {
            return GetSubType(typeof(Optimizer));
        }

        /// <summary>
        ///     Get All Loss Function
        /// </summary>
        /// <returns></returns>
        public Type[] GetLossTypes()
        {
            return GetSubType(typeof(Loss));
        }

        /// <summary>
        ///     Get All Metric Type
        /// </summary>
        /// <returns></returns>
        public Type[] GetMerticTypes()
        {
            return GetSubType(typeof(Metric));
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