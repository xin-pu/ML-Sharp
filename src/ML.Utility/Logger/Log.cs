using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;

namespace ML.Utility.Logger
{
    public class Log
    {
        public static Dictionary<int, RichLog> TextBoxManager = new Dictionary<int, RichLog>();
        public static int defaultID = 0;

        private static Log _logInstance;
        private static readonly object Locker = new object();


        static Log()
        {
        }


        public static Log GetInstance()
        {
            lock (Locker)
            {
                if (_logInstance == null) _logInstance = new Log();
            }

            return _logInstance;
        }

        public void Dispose()
        {
        }


        public static void UnRegisterLog(int threadID)
        {
            if (!TextBoxManager.Keys.Contains(threadID)) return;
            TextBoxManager.Remove(threadID);
        }

        public static void AddLog(int threadID, LogType logtype = LogType.Message, object msg = null)
        {
            if (!TextBoxManager.Keys.Contains(threadID)) return;
            TextBoxManager[threadID]?.Print(logtype, msg);
        }

        public static void AddLog(LogType logtype = LogType.Message, object msg = null)
        {
            var threadID = defaultID;
            AddLog(threadID, logtype, msg);
        }

        [Obsolete("please replace with MultiLog.AddLog")]
        public static void AllAddLog(LogType logtype = LogType.Message, object msg = null)
        {
            var id = Thread.CurrentThread.ManagedThreadId;
            MultiLog.AddLog(id, logtype, msg);
        }


        #region Operation

        public static void ClearLog(int threadID)
        {
            if (!TextBoxManager.Keys.Contains(threadID)) return;
            TextBoxManager[threadID]?.Clear();
        }

        public static void SaveLog(int threadID, string path)
        {
            if (!TextBoxManager.Keys.Contains(threadID)) return;
            TextBoxManager[threadID]?.SaveTxtLog(path);
        }

        public static void SaveRtf(int threadID, string path)
        {
            if (!TextBoxManager.Keys.Contains(threadID)) return;
            TextBoxManager[threadID]?.SaveRtfLog(path);
        }


        public static void SaveRtf(RichTextBox richTextBox, string path)
        {
            var file = new FileStream(path, FileMode.Create);
            var textRange = new TextRange(
                richTextBox.Document.ContentStart,
                richTextBox.Document.ContentEnd
            );
            textRange.Save(file, DataFormats.Rtf);
            file.Close();
        }

        #endregion
    }
}