using System;
using System.Collections.Generic;
using System.Linq;
using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utility;
using Numpy;

namespace ML.Core.Models
{
    public class Perceptron<T> : ModelGD<T>
        where T : DataView
    {
        /// <summary>
        ///     Y should be [Batch, One Hot]
        ///     [
        ///     [0,0,1],
        ///     [0,1,0]
        ///     ]
        /// </summary>
        /// <param name="classes"></param>
        public Perceptron(int classes)
        {
            Transformer = new LinearFirstorder();
            Classes = classes;
        }


        public int Classes { set; get; }
        public override string Description { get; }

        public Dictionary<int, NDarray> WeightsDict { set; get; }
        public Dictionary<int, Variable[]> VariablesDict { set; get; }

        internal void InitialVariables()
        {
            VariablesDict = Variables
                .Select((v, i) => (i / Classes, v))
                .GroupBy(p => p.Item1)
                .ToDictionary(
                    p => p.Key,
                    p => p.Select(a => a.v).ToArray());
        }

        //public override Term[] CallGraph(NDarray features)
        //{
        //    var lablesPredict = VariablesDict
        //        .ToDictionary(
        //            p => p.Key,
        //            p =>
        //            {
        //                var feature = Transformer.Call(features);
        //                return term.matmul(p.ValueError, feature);
        //            });

        //    throw new NotImplementedException();
        //}

        public override TermMatrix CallGraph(NDarray features)
        {
            throw new NotImplementedException();
        }

        public override NDarray Call(NDarray features)
        {
            var feature = Transformer.Call(features);
            var y_pred = nn.sigmoid(np.matmul(feature, Weights.T));
            return sign(y_pred);
        }

        private NDarray sign(NDarray inputDarray)
        {
            return np.expand_dims(np.argmax(inputDarray, -1), -1);
        }
    }
}