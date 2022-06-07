using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Trainers
{
    public class GDTrainer<T> : MvxViewModel
        where T : DataView
    {
        private Loss _loss;

        private ObservableCollection<Metric> _metrics;
        private IModelGD<T> _modelGd;
        private Optimizer _optimizer;
        private Dataset<T> _trainDataset;

        private TrainPlan _trainPlan;
        private Dataset<T> _valDataset;

        public Action<GDTrainer<T>> AfterBatchPipeline;
        public Action<GDTrainer<T>> AfterEpochPipeline;

        public Action<GDTrainer<T>> BeforeBatchPipeline;
        public Action<GDTrainer<T>> BeforeEpochPipeline;
        public Action<string> Print;


        public IModelGD<T> ModelGd
        {
            get => _modelGd;
            set => SetProperty(ref _modelGd, value);
        }

        public Dataset<T> TrainDataset
        {
            get => _trainDataset;
            set => SetProperty(ref _trainDataset, value);
        }

        public Dataset<T> ValDataset
        {
            get => _valDataset;
            set => SetProperty(ref _valDataset, value);
        }

        public Optimizer Optimizer
        {
            get => _optimizer;
            set => SetProperty(ref _optimizer, value);
        }

        public Loss Loss
        {
            get => _loss;
            set => SetProperty(ref _loss, value);
        }

        public ObservableCollection<Metric> Metrics
        {
            get => _metrics;
            set => SetProperty(ref _metrics, value);
        }

        public TrainPlan TrainPlan
        {
            get => _trainPlan;
            set => SetProperty(ref _trainPlan, value);
        }

        public async Task Fit()
        {
            TrainDataset.Should().NotBeNull("dataset should not ne null");

            ModelGd.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);

                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<T> data)
                    {
                        if (data.Count == 0)
                            continue;

                        BeforeBatchPipeline?.Invoke(this);
                        var batchdataSet = data.ToDatasetNDarray();

                        var predTerms = ModelGd.CallGraph(batchdataSet.Feature);
                        var lossTerm = Loss.GetLossTerm(predTerms, batchdataSet.Label, ModelGd.Variables);


                        NDarray GetGradient(NDarray weight)
                        {
                            var gradientArray = lossTerm.Differentiate(ModelGd.Variables, weight.GetData<double>());
                            var g = np.array(gradientArray);
                            return np.reshape(g, weight.shape);
                        }

                        ModelGd.Weights = Optimizer.Call(ModelGd.Weights, GetGradient, e);

                        AfterBatchPipeline?.Invoke(this);
                    }

                    var trainMsg = new StringBuilder($"#{e + 1:D4}\t");
                    var train_loss = UpdateLossMetric(TrainDataset);
                    trainMsg.Append($"[Loss]:{train_loss:F4}\t");
                    foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");

                    if (ValDataset != null)
                    {
                        var val_loss = UpdateLossMetric(ValDataset);
                        trainMsg.Append("\tVal\t");
                        trainMsg.Append($"[Loss]-V:{val_loss:F4}\t");
                        foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");
                    }

                    Print?.Invoke(trainMsg.ToString());

                    AfterEpochPipeline?.Invoke(this);
                });

            /// early stoping
            /// Print status of each epoch
        }

        public double UpdateLossMetric(Dataset<T> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var y_pred = ModelGd.Call(dataview.Feature);
            var y_true = dataview.Label;
            var predterms = ModelGd.CallGraph(dataview.Feature);
            var lossTerm = Loss.GetLossTerm(predterms, dataview.Label, ModelGd.Variables);
            var loss = lossTerm.Evaluate(ModelGd.Variables, ModelGd.GetWeightArray());
            Metrics.ToList().ForEach(m => m.Call(y_pred, y_true));

            return loss;
        }
    }
}