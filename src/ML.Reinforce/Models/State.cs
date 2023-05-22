﻿using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Reinforce.Models
{
    /// <summary>
    ///     环境的状态
    /// </summary>
    public class State : ObservableObject
    {
        private DateTime _timeStamp;
        private NDarray _value = np.empty();

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }


        /// <summary>
        ///     奖励的张量格式
        /// </summary>
        public NDarray Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{Value}";
        }
    }
}