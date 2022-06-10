using System.ComponentModel;
using GalaSoft.MvvmLight;

namespace ML.Core.Trainers
{
    public class TrainPlan : ViewModelBase
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
            set => Set(ref _epoch, value);
        }

        [Category("Configuration")]
        public int BatchSize
        {
            get => _batchSize;
            set => Set(ref _batchSize, value);
        }
    }
}