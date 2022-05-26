using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using FluentAssertions;

namespace ML.Core.Data.Loader
{
    /// <summary>
    ///     A text loader for get IEnum<T> from txt file.
    /// </summary>
    public class TextLoader
    {
        public static DataSet<T> LoadDataSet<T>(string path, bool hasHeader = true, char splitChar = ',')
        {
            /// Step 0 Precheck
            File.Exists(path).Should().BeTrue($"File {path} should exist.");

            /// Step 1 Read Stream to DataTable or Array
            using var stream = new StreamReader(path);
            var allline = stream.ReadToEnd()
                .Split('\r', '\n')
                .Where(a => !string.IsNullOrEmpty(a))
                .ToList();
            if (hasHeader)
                allline.RemoveAt(0);
            var alldata = allline.Select(l => l.Split(splitChar).ToArray()).ToList();

            /// Step 2 Get Field Dict which have LoadColumn
            var fieldDict = GetFieldDict(typeof(T));

            /// Step 2 According LoadColumnAttribute Change to Data
            var datas = alldata.Select(single => GetData<T>(fieldDict, single)).ToList();

            /// Step 3 Return DataSet 
            return new DataSet<T>(datas);
        }

        private static Dictionary<FieldInfo, int> GetFieldDict(Type type)
        {
            var fieldInfo = type.GetFields()
                .Where(a => a.CustomAttributes
                    .Any(attributeData => attributeData.AttributeType == typeof(LoadColumnAttribute)))
                .ToList();
            var dict = fieldInfo.ToDictionary(
                f => f,
                f =>
                {
                    var aa = f.GetCustomAttribute<LoadColumnAttribute>();
                    return aa.ColumnIndex;
                });
            return dict;
        }

        private static T GetData<T>(Dictionary<FieldInfo, int> dict, string[] array)
        {
            var obj = Activator.CreateInstance(typeof(T));
            dict.ToList().ForEach(p =>
            {
                var fieldInfo = p.Key;
                var strValue = array[p.Value];
                var type = fieldInfo.FieldType;

                if (type == typeof(string))
                    fieldInfo.SetValue(obj, strValue);
                else if (type == typeof(int))
                    fieldInfo.SetValue(obj, int.Parse(strValue));
                else if (type == typeof(long))
                    fieldInfo.SetValue(obj, long.Parse(strValue));
                else if (type == typeof(double))
                    fieldInfo.SetValue(obj, double.Parse(strValue));
                else if (type == typeof(float)) fieldInfo.SetValue(obj, float.Parse(strValue));
                else if (type == typeof(byte)) fieldInfo.SetValue(obj, byte.Parse(strValue));
                else
                    fieldInfo.SetValue(obj, null);
            });

            return (T) obj;
        }
    }
}