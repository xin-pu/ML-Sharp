using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ML.Core.Trainers
{
    public class TrainPlan : ObservableObject
    {
        private int _batchSize;
        private int _epoch;

        public TrainPlan()
        {
            BatchSize = 0;
            Epoch = 10;
        }


        [Category("Configuration")]
        public int Epoch
        {
            get => _epoch;
            set => SetProperty(ref _epoch, value);
        }

        [Category("Configuration")]
        public int BatchSize
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }
    }
}