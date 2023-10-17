# Hindi_2_Chhattisgarhi_Translation
HINDI tO CHHATTISGARHI machine translation using LSTM 
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNSaxDJddWZDusrvy3GhAma",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Pankajkashyap1/Hindi_2_Chhattisgarhi_Translation/blob/master/h2cg_using_LSTM.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "6fzIp2gpnQ02"
      },
      "outputs": [],
      "source": [
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "\n",
        "# Input data files are available in the \"../input/\" directory.\n",
        "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
        "\n",
        "import os\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "import keras\n",
        "from keras.models import Model\n",
        "from keras.layers import Input, LSTM, Dense,TimeDistributed,Embedding,Bidirectional\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing.sequence import pad_sequences\n",
        "from string import digits\n",
        "import nltk\n",
        "import re\n",
        "import string\n",
        "pd.set_option('display.max_rows', 500)\n",
        "pd.set_option('display.max_columns', 500)\n",
        "pd.set_option('display.width', 1000)\n",
        "pd.set_option('display.max_colwidth', -1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CshLDrAVnRWk",
        "outputId": "33c17706-fbfe-467e-b4a7-0f2995d7b6a3"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-2-114039e89046>:16: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.\n",
            "  pd.set_option('display.max_colwidth', -1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "uploaded= files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 69
        },
        "id": "fuGnddUCnRrp",
        "outputId": "4400364a-4ccf-4d82-ba66-e41437327c94"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-6aa459cf-fed5-4c61-bb4e-a0f20b45ec42\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-6aa459cf-fed5-4c61-bb4e-a0f20b45ec42\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving work.xlsx to work (2).xlsx\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lines=pd.read_excel(\"work.xlsx\")\n",
        "lines = lines[:256]\n",
        "lines.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "l7EnOzcNnRnb",
        "outputId": "c6ba0fc7-71bf-4f14-cb40-d66c244172b0"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                   cg_sentance                                                              hindi_sentance\n",
              "0  नहाय के बाद सरीर ल बने अंगोछना चाही          नहाने के बाद शरीर को अच्छे से पोंछना चाहिए                                \n",
              "1  अंगरा के सेके रोटी ल अंगाकर कथे              अंगर से सेके रोटी को चावल की मोती रोटी कहते हैं                           \n",
              "2  मोर दाई ह अँचरा म पइसा गठियाय हे             मेरी मां ने सिक्कों को अपने आँचल में बांध रखा है                          \n",
              "3  अंचावन ल अनदेखी करबे त खटिया ढिल्ला हो जाही  खाट की रस्सियों को कसने की रस्सी को नज़रअंदाज़ करेंगे तो खाट ढीली हो जाएगी\n",
              "4  दिन के अंजोर म रद्दा नइ भूला हूँ             दिन के उजाले में रास्ते से नहीं भटकूंगा                                   "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9f4d2d6d-d01e-4a19-b1d8-508b09e878ad\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>cg_sentance</th>\n",
              "      <th>hindi_sentance</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>नहाय के बाद सरीर ल बने अंगोछना चाही</td>\n",
              "      <td>नहाने के बाद शरीर को अच्छे से पोंछना चाहिए</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>अंगरा के सेके रोटी ल अंगाकर कथे</td>\n",
              "      <td>अंगर से सेके रोटी को चावल की मोती रोटी कहते हैं</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>मोर दाई ह अँचरा म पइसा गठियाय हे</td>\n",
              "      <td>मेरी मां ने सिक्कों को अपने आँचल में बांध रखा है</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>अंचावन ल अनदेखी करबे त खटिया ढिल्ला हो जाही</td>\n",
              "      <td>खाट की रस्सियों को कसने की रस्सी को नज़रअंदाज़ करेंगे तो खाट ढीली हो जाएगी</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>दिन के अंजोर म रद्दा नइ भूला हूँ</td>\n",
              "      <td>दिन के उजाले में रास्ते से नहीं भटकूंगा</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "      \n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9f4d2d6d-d01e-4a19-b1d8-508b09e878ad')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "      \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-9f4d2d6d-d01e-4a19-b1d8-508b09e878ad button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-9f4d2d6d-d01e-4a19-b1d8-508b09e878ad');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "  \n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Lowercase all characters\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: str(x))\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: str(x))\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: x.lower())\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: x.lower())"
      ],
      "metadata": {
        "id": "w-CHXcjbnRvp"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Remove quotes\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: re.sub(\"'\", '', x))\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: re.sub(\"'\", '', x))"
      ],
      "metadata": {
        "id": "Una27oJ3nRze"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "exclude = set(string.punctuation) # Set of all special characters\n",
        "# Remove all the special characters\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))"
      ],
      "metadata": {
        "id": "ToXtW07-nR22"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Remove all numbers from text\n",
        "remove_digits = str.maketrans('', '', digits)\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: x.translate(remove_digits))\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: x.translate(remove_digits))\n",
        "\n",
        "lines['cg_sentance'] = lines['cg_sentance'].apply(lambda x: re.sub(\"[२३०८१५७९४६]\", \"\", x))\n",
        "\n",
        "# Remove extra spaces\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: x.strip())\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: x.strip())\n",
        "lines['hindi_sentance']=lines['hindi_sentance'].apply(lambda x: re.sub(\" +\", \" \", x))\n",
        "lines['cg_sentance']=lines['cg_sentance'].apply(lambda x: re.sub(\" +\", \" \", x))"
      ],
      "metadata": {
        "id": "wxq0GjGpnR6G"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Add start and end tokens to target sequences\n",
        "lines['cg_sentance'] = lines['cg_sentance'].apply(lambda x : 'START_ '+ x + ' _END')"
      ],
      "metadata": {
        "id": "0yWhSIqdnR9a"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### Get Hindi and CG Vocabulary\n",
        "all_hindi_words=set()\n",
        "for hindi in lines['hindi_sentance']:\n",
        "    for word in hindi.split():\n",
        "        if word not in all_hindi_words:\n",
        "            all_hindi_words.add(word)\n",
        "\n",
        "all_cg_words=set()\n",
        "for cg in lines['cg_sentance']:\n",
        "    for word in cg.split():\n",
        "        if word not in all_cg_words:\n",
        "            all_cg_words.add(word)"
      ],
      "metadata": {
        "id": "jKA5yzovnSEj"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lines['length_hindi_sentance']=lines['hindi_sentance'].apply(lambda x:len(x.split(\" \")))\n",
        "lines['length_cg_sentance']=lines['cg_sentance'].apply(lambda x:len(x.split(\" \")))"
      ],
      "metadata": {
        "id": "IYocP2bunSIE"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lines.head()\n",
        "lines[lines['length_hindi_sentance']>30].shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E_Oz5_FdnSLd",
        "outputId": "ac904a3a-a72b-471f-a670-86ccc4a9ec2f"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(0, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lines=lines[lines['length_hindi_sentance']<=20]\n",
        "lines=lines[lines['length_cg_sentance']<=20]"
      ],
      "metadata": {
        "id": "nct3RZ2XnSO1"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"maximum length of CG Sentence \",max(lines['length_cg_sentance']))\n",
        "print(\"maximum length of HINDI Sentence \",max(lines['length_hindi_sentance']))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MtkBGkFPnSSX",
        "outputId": "07387ea3-58c0-461f-d799-a0b2a8f14e05"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "maximum length of CG Sentence  16\n",
            "maximum length of HINDI Sentence  16\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "max_length_src=max(lines['length_cg_sentance'])\n",
        "max_length_tar=max(lines['length_hindi_sentance'])"
      ],
      "metadata": {
        "id": "-_o4bXt2nSV4"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_words = sorted(list(all_hindi_words))\n",
        "target_words = sorted(list(all_cg_words))\n",
        "num_encoder_tokens = len(all_hindi_words)\n",
        "num_decoder_tokens = len(all_cg_words)\n",
        "num_encoder_tokens, num_decoder_tokens"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "578-jU7DnSZf",
        "outputId": "1e76b942-cda1-4eac-85b2-c726578ac3d6"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(749, 793)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "num_decoder_tokens += 1"
      ],
      "metadata": {
        "id": "EAI9TTBBnSc7"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])\n",
        "target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])"
      ],
      "metadata": {
        "id": "tELQz_C1nShC"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())\n",
        "reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())"
      ],
      "metadata": {
        "id": "bk-xd5ACnSkf"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lines.head(10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 427
        },
        "id": "qL6NNaL0nSoI",
        "outputId": "a23e3fce-0832-432e-cb8a-f277384b8c45"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                               cg_sentance                                                              hindi_sentance  length_hindi_sentance  length_cg_sentance\n",
              "0  START_ नहाय के बाद सरीर ल बने अंगोछना चाही _END          नहाने के बाद शरीर को अच्छे से पोंछना चाहिए                                  9                      10                \n",
              "1  START_ अंगरा के सेके रोटी ल अंगाकर कथे _END              अंगर से सेके रोटी को चावल की मोती रोटी कहते हैं                             11                     9                 \n",
              "2  START_ मोर दाई ह अँचरा म पइसा गठियाय हे _END             मेरी मां ने सिक्कों को अपने आँचल में बांध रखा है                            11                     10                \n",
              "3  START_ अंचावन ल अनदेखी करबे त खटिया ढिल्ला हो जाही _END  खाट की रस्सियों को कसने की रस्सी को नज़रअंदाज़ करेंगे तो खाट ढीली हो जाएगी  15                     11                \n",
              "4  START_ दिन के अंजोर म रद्दा नइ भूला हूँ _END             दिन के उजाले में रास्ते से नहीं भटकूंगा                                     8                      10                \n",
              "5  START_ बने अंजोर म पढ़ना लिखना चाही _END                 उचित प्रकाश में पढ़ना लिखना करना चाहिए                                      7                      8                 \n",
              "6  START_ हमन ल अंजोरी रात म चले म मजा आइस _END             हमें चांदनी रात में घूमने में मज़ा आया                                      8                      11                \n",
              "7  START_ अण्डी तेल ह मालिस बर अच्छा होथे _END              अरंडी का तेल मालिश के लिए अच्छा होता है                                     9                      9                 \n",
              "8  START_ अर्जुन ह पांडव मन म अंतरमंझिला रिहिस _END         पांडवों में अर्जुन पाँच भाइयों में तीसरे थे                                 8                      9                 \n",
              "9  START_ गणित म अंताजी नइ चलय _END                         गणित में अनुमान नही चलता                                                    5                      7                 "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fcb1818b-2dde-46e5-b031-f59140aae311\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>cg_sentance</th>\n",
              "      <th>hindi_sentance</th>\n",
              "      <th>length_hindi_sentance</th>\n",
              "      <th>length_cg_sentance</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>START_ नहाय के बाद सरीर ल बने अंगोछना चाही _END</td>\n",
              "      <td>नहाने के बाद शरीर को अच्छे से पोंछना चाहिए</td>\n",
              "      <td>9</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>START_ अंगरा के सेके रोटी ल अंगाकर कथे _END</td>\n",
              "      <td>अंगर से सेके रोटी को चावल की मोती रोटी कहते हैं</td>\n",
              "      <td>11</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>START_ मोर दाई ह अँचरा म पइसा गठियाय हे _END</td>\n",
              "      <td>मेरी मां ने सिक्कों को अपने आँचल में बांध रखा है</td>\n",
              "      <td>11</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>START_ अंचावन ल अनदेखी करबे त खटिया ढिल्ला हो जाही _END</td>\n",
              "      <td>खाट की रस्सियों को कसने की रस्सी को नज़रअंदाज़ करेंगे तो खाट ढीली हो जाएगी</td>\n",
              "      <td>15</td>\n",
              "      <td>11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>START_ दिन के अंजोर म रद्दा नइ भूला हूँ _END</td>\n",
              "      <td>दिन के उजाले में रास्ते से नहीं भटकूंगा</td>\n",
              "      <td>8</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>START_ बने अंजोर म पढ़ना लिखना चाही _END</td>\n",
              "      <td>उचित प्रकाश में पढ़ना लिखना करना चाहिए</td>\n",
              "      <td>7</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>START_ हमन ल अंजोरी रात म चले म मजा आइस _END</td>\n",
              "      <td>हमें चांदनी रात में घूमने में मज़ा आया</td>\n",
              "      <td>8</td>\n",
              "      <td>11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>START_ अण्डी तेल ह मालिस बर अच्छा होथे _END</td>\n",
              "      <td>अरंडी का तेल मालिश के लिए अच्छा होता है</td>\n",
              "      <td>9</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>START_ अर्जुन ह पांडव मन म अंतरमंझिला रिहिस _END</td>\n",
              "      <td>पांडवों में अर्जुन पाँच भाइयों में तीसरे थे</td>\n",
              "      <td>8</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>START_ गणित म अंताजी नइ चलय _END</td>\n",
              "      <td>गणित में अनुमान नही चलता</td>\n",
              "      <td>5</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "      \n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fcb1818b-2dde-46e5-b031-f59140aae311')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "      \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-fcb1818b-2dde-46e5-b031-f59140aae311 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-fcb1818b-2dde-46e5-b031-f59140aae311');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "  \n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X, y = lines['hindi_sentance'], lines['cg_sentance']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)\n",
        "X_train.shape, X_test.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uNSVkzAXnSry",
        "outputId": "c46e7f33-2092-414b-d345-664bbd7e401a"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((204,), (52,))"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zJiMeObGnSvX",
        "outputId": "7ecb46a2-0dd5-40d2-ba6c-afa70cae6ad9"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "143    तुमने लिखा                                                                \n",
              "84     हरिया का घर यहीं से सीधा है कहीं और मत जाना                               \n",
              "55     मुसीबत में काम आये वही मित्र है                                           \n",
              "220    अलवा जलवा हलवा भी स्वादिष्ठ लगता है                                       \n",
              "104    मेरे छोटे बेटे का हाथ पंखे के बटन तक नहीं पहुँचता                         \n",
              "139    वे दो घंटे से रामायण पढ़ रहे हैं                                          \n",
              "127    तुम खा रहे हो                                                             \n",
              "60     जिद्दी लड़का किसी की नहीं सुनता                                           \n",
              "101    यह किताब उस बहन तक पहुँचा देना                                            \n",
              "172    सीता पढ़ेगी                                                               \n",
              "245    फूल मुरझा गया है                                                          \n",
              "126    हम खा रहे हैं                                                             \n",
              "225    जो गाली देगा उसे भी एक मुक्का जोर से मार देना चाहिए                       \n",
              "144    सीता ने लिखा                                                              \n",
              "108    बहुत सारे सब्जी भाजी और लताएँ बिना बोए उग जाते हैं                        \n",
              "178    रात में कोटवार चिल्लाते और पहरा देते हैं                                  \n",
              "73     कटहल का अचार बनाना जानते हैं                                              \n",
              "114    पता नहीं मेरा बेटा कहां फंसा है                                           \n",
              "158    मैं पढ़ चुका रहूँगा                                                       \n",
              "69     शरारती बच्चे परीक्षा के दिन रोते हैं                                      \n",
              "141    हमने लिखा                                                                 \n",
              "109    छोटे रास्ते से जाएंगे तो जल्द ही पहुंच जाएंगे                             \n",
              "115    बहुत रात हो गई अभी तक घर नहीं आया है                                      \n",
              "246    मजदूर आज दिन रहते चले गए                                                  \n",
              "113    रमेसर कितनो से उलझ जाता है                                                \n",
              "243    सीधे जाओ                                                                  \n",
              "90     कहीं भी देर से जाना ठीक बात नहीं होता                                     \n",
              "29     अकरी की दाल सेहत के लिए अच्छी नहीं होती है                                \n",
              "170    हम पढ़ेंगे                                                                \n",
              "82     दोस्त अधिकारी बनने पर अजनबी जैसे मिलाता है                                \n",
              "111    आवेदन यहां नवंबर तक पहुंच जाना चाहिए                                      \n",
              "5      उचित प्रकाश में पढ़ना लिखना करना चाहिए                                    \n",
              "56     अटर्रा नींबू से बड़ा होता है                                              \n",
              "132    तुम खा चुके हो                                                            \n",
              "154    हम पढते रहेंगे                                                            \n",
              "162    वे पढ़ चुके रहेंगे                                                        \n",
              "65     इतना सा प्रसाद सबके लिए नहीं होगा                                         \n",
              "186    हार के बाद सेना पीछे हट गई                                                \n",
              "85     वह अनपढ़ है लेकिन वह समझदारी से बात करता है                               \n",
              "219    दादा के पैर को धीरे धीरे दबाना चाहिए                                      \n",
              "237    सेब सेहत के लिए अच्छे होते हैं                                            \n",
              "31     एकल बैल के लिए जोड़ी खोजने की जरूरत है                                    \n",
              "12     चावल पकाने के लिए पानी डालें                                              \n",
              "35     एक भविष्यवक्ता ने मुझसे कहा कि मैं किसी दिन नेता बनूंगा                   \n",
              "28     पहली बारिश के बाद खेत की जुताई से सभी खरपतवार सूख गए                      \n",
              "42     ट्रेन के इंतजार में थक गया                                                \n",
              "112    मेरी गेंद पेड़ की टहनी में फंस गई है                                      \n",
              "22     पिता अपने बेटे के लिए खिलौना और मिठाई लाए                                 \n",
              "125    मैं खा रहा हूँ                                                            \n",
              "93     बहुत देर हो चुकी है लड़का अभी तक नहीं उठा है                              \n",
              "173    वे पढ़ेंगे                                                                \n",
              "251    गाय के गले में सोहाई बंधा है                                              \n",
              "51     आज रात गांव में नाचा होगा                                                 \n",
              "240    बच्चे को सोखना सोखा दो                                                    \n",
              "95     छाछ नहीं है तो सूखे आम का चूर्ण डाल कर सब्जी पका लेना                     \n",
              "146    हम लिख रहे थे                                                             \n",
              "204    सरपंच को डर नहीं है निरंकुश गाँव चला रहा है                               \n",
              "76     मेरा काम अधर में लटक गया है                                               \n",
              "41     किसान धान की बिजाई के लिए बारिश का इंतजार करता                            \n",
              "119    ज्यादा आराम करने से आलस आता है                                            \n",
              "155    तुम पढ़ते रहोगे                                                           \n",
              "78     बटाईदार अब हमारी खेती में काम कर रहे हैं                                  \n",
              "150    वे लिख रहे थे                                                             \n",
              "26     रजाई से सिर को न डाक घुटन महसूस होगा                                      \n",
              "247    पर काम में कामचोर                                                         \n",
              "168    दिवाली के दिन राउत महिलाएं अपने मालिक के घर की दीवार पर हंथा छापती हैं    \n",
              "118    अलसी का तेल घावों के लिए बहुत काम आता है                                  \n",
              "193    कुदाली चलाने के कारण रामलाल के हाथ में गठान पड़ गया                       \n",
              "140    मैंने लिखा                                                                \n",
              "0      नहाने के बाद शरीर को अच्छे से पोंछना चाहिए                                \n",
              "2      मेरी मां ने सिक्कों को अपने आँचल में बांध रखा है                          \n",
              "77     हमारा देश आधी रात को स्वतंत्र हुआ                                         \n",
              "46     मनुष्य चाँद पर पहुँच गया है                                               \n",
              "100    हम कल बजे रायपुर पहुंचेंगे                                                \n",
              "205    कमाने वाला पतला हो रहा है                                                 \n",
              "159    हम पढ़ चुके रहेंगे                                                        \n",
              "183    शाम तक हम रायपुर पहुंच जाएंगे                                             \n",
              "254    आज बाजार से दो सूप खरीदा                                                  \n",
              "98     पहले हमारे गाँव में आम का बड़ा बाग था                                     \n",
              "36     कार्तिक मास में हम आकाश के दीपक                                           \n",
              "61     वह अपनी माँ के यहाँ गई है                                                 \n",
              "200    कोठी से धान निकालते निकालते नानी गिर गई                                   \n",
              "142    तुमने लिखा                                                                \n",
              "11     कैसे मिलेगा                                                               \n",
              "250    जल्दी करो                                                                 \n",
              "224    हकलाने वाला भी एक दिन ठीक हो जाता है                                      \n",
              "27     पहली बारिश होते ही मैंने खेत की जुताई कर दी                               \n",
              "231    देख कर नहीं चलोगे तो ठोकर खा जाओगे                                        \n",
              "4      दिन के उजाले में रास्ते से नहीं भटकूंगा                                   \n",
              "122    तुम खाते हो                                                               \n",
              "32     घर में अकेले रहने पर मुझे डर नहीं लगता                                    \n",
              "147    मैं लिख रहा था                                                            \n",
              "182    बारिश में घर की छत एकाएक गिर गई                                           \n",
              "138    आप एक घंटे से खड़े हैं                                                    \n",
              "62     आठ दिन बिना नहीं आएंगे                                                    \n",
              "135    तीन घंटे से बारिश हो रही है                                               \n",
              "128    सीता खा रही है                                                            \n",
              "232    जंगल में अकेला पाकर भेड़िया काट देता है                                   \n",
              "194    हर्रा बहेड़ा और आंवला मिलाने से पाचक चूर्ण जाता है                        \n",
              "70     रात के समय उल्लू का चिल्लाना बहुत डरावना लगता है                          \n",
              "197    नहीं तो बस छूट जाएगी                                                      \n",
              "64     नहीं खा पाएंगे                                                            \n",
              "44     समोसा खा खा कर तृप्त हो गए हैं                                            \n",
              "165    मैं आठ घंटे से पढ़ रहा हूँगा                                              \n",
              "156    सीता पढ़ती रहेगी                                                          \n",
              "40     कछुआ ने खरगोश को पछाड़ दिया                                               \n",
              "123    वे खाते हैं                                                               \n",
              "153    मैं पढ़ता रहूँगा                                                          \n",
              "23     इतने पैसे में काम नहीं चलेगा                                              \n",
              "192    इस साल अच्छी फसल हुई तो सुधू ने अपनी पत्नी के लिए चांदी का कड़ा खरीदा     \n",
              "249    और लगेगा                                                                  \n",
              "81     अजनबी अलग से पहचाना में आ जाता है                                         \n",
              "39     दादाजी हर छोटीछोटी बातों में उग्र हो जाते हैं                             \n",
              "244    जल्दी पहुंचोगे                                                            \n",
              "47     फिर भी यह आश्चर्य नहीं है                                                 \n",
              "94     फालतू की बात मत करो                                                       \n",
              "195    विवाह में दूल्हा दुल्हन के शरीर पर हल्दी लगाया जाता है                    \n",
              "161    सीता पढ़ चुकी रहेगी                                                       \n",
              "43     ज्यादा खाना ठीक नहीं है                                                   \n",
              "145    उन्होंने लिखा                                                             \n",
              "175    कपिल ने अपने बच्चों को उनके हिस्सा बाँटा दे दिए                           \n",
              "3      खाट की रस्सियों को कसने की रस्सी को नज़रअंदाज़ करेंगे तो खाट ढीली हो जाएगी\n",
              "105    सत्य वचन अमृत के समान होते हैं                                            \n",
              "53     कोई दवा कारगर नहीं है                                                     \n",
              "133    सीता खा चुकी है                                                           \n",
              "233    हम मिल जुल कर छत्तीसगढ़ में रहते हैं                                      \n",
              "198    कोटवार ने बैठक के लिए गांव में जोर की आवाज लगाई                           \n",
              "238    हाथी की बड़ी सी सूंड होती है                                              \n",
              "49     खाना खाने के बाद हाथ को अच्छे से धोएं                                     \n",
              "163    तीन घंटे से बारिश हो रही होगी                                             \n",
              "80     छत्तीसगढ़ में अनाज को अन्ना कुंवारी माना जाता है                          \n",
              "34     उसे अपनी इस हरकत पर बुरा लगा                                              \n",
              "211    चिमटी से चुभे हुए कांटे को निकालते है                                     \n",
              "7      अरंडी का तेल मालिश के लिए अच्छा होता है                                   \n",
              "171    तुम पढ़ोगे                                                                \n",
              "216    लड़ाके से लड़ाइ करके लड़की पेंसिल की नोक में कोंच दी                      \n",
              "110    अरगनी में कपड़े को टांग दो                                                \n",
              "91     अभी गांव से आया हूं                                                       \n",
              "83     मेरा पड़ोसी बहुत ईर्ष्यालु व्यक्ति है                                     \n",
              "229    जर्मन के बर्तन हल्के होते हैं                                             \n",
              "234    हमारी खुद की ताकत हमारी काम आएगी                                          \n",
              "89     भांजी की शादी में बहुत ज्यादा पैसा लग गया                                 \n",
              "8      पांडवों में अर्जुन पाँच भाइयों में तीसरे थे                               \n",
              "13     अंधना तैयार है                                                            \n",
              "59     वह बारबार अंगड़ाई लेता है                                                 \n",
              "221    भुखखड़ लड़के का पेट दर्द करेगा ही                                         \n",
              "131    हम खा चुके हैं                                                            \n",
              "17     विटामिन ए की कमी से रतौंधी हो जाती है                                     \n",
              "166    आप एक घंटे से खड़े होंगे                                                  \n",
              "72     पेट दर्द होगा                                                             \n",
              "226    जोर से मुक्का खाएगा तो लड़का गाली देना छोड़ ही देगा                       \n",
              "134    वे खा चुके हैं                                                            \n",
              "209    पढ़ने वाले लड़कों को बराबरी का दुराग्रह नहीं करना चाहिए                   \n",
              "236    कमजोर बछड़े को कुत्ता भी काट देता है                                      \n",
              "63     बच्चे को इतना चावल मत देना                                                \n",
              "54     गणित में तीर तुक्का नहीं चलता                                             \n",
              "107    खट्टी करी की सब्जी बहुत ही स्वादिष्ट होती है                              \n",
              "50     हवाई जहाज देखते देखते गायब हो गया                                         \n",
              "212    खराब अक्षर को शिक्षक भी अवमूल्यन करते हैं                                 \n",
              "174    जो हार गया वह दूर जा सकता है                                              \n",
              "213    नाटक देखना है तो हल्ला मत करो                                             \n",
              "189    होनी को कोई नहीं टाल सकता                                                 \n",
              "252    कुरता को सुईधागे से सिल दो                                                \n",
              "207    किसान ने हाय हाय करके कमाया फिर भी पड़ गया अकाल                           \n",
              "227    जो कुत्ता काटने को दौड़े उसे पत्थर मार दे                                 \n",
              "169    मैं पढूँगा                                                                \n",
              "58     बाधा डालने वाले किसी कार्य को पूर्ण नहीं करने देते                        \n",
              "218    पहले वे खेतों में रहते थे                                                 \n",
              "48     बच्चे लोग जिराफ को देखकर हैरान रह गए                                      \n",
              "88     बूढ़ा बहुत जल्द मरने वाला है                                              \n",
              "21     मेरे पास भी इसी तरह का टेलीविजन है                                        \n",
              "57     भारी पत्थर को उत्तोलक से बाहर निकालें                                     \n",
              "203    खाने में रंग धंग नहीं है                                                  \n",
              "160    तुम पढ़ चुके रहोगे                                                         \n",
              "248    और नहीं खाएंगे                                                            \n",
              "187    कुश्ती में राकेश रोहित से हार गया                                         \n",
              "191    तोता मैगपाई एक ही रंग के होते हैं                                         \n",
              "129    वे खा रहे हैं                                                             \n",
              "37     मेरी जांघ पर फोड़ा हो गया है                                              \n",
              "157    वे पढ़ते रहेंगे                                                           \n",
              "241    यह प्रश्न साधारण है                                                       \n",
              "1      अंगर से सेके रोटी को चावल की मोती रोटी कहते हैं                           \n",
              "52     भगवान जाने ये कैसी बीमारी है                                              \n",
              "149    सीता ने लिखा                                                              \n",
              "130    मैं खा चुका हूँ                                                           \n",
              "151    हम लिख चुके थे                                                            \n",
              "103    मैं अपनी बहन को उसके ससुराल पहुँचाने गया था                               \n",
              "99     हमें पांच बजे तक पहुंचना है                                               \n",
              "116    आजकल पपीते की खेती की बहुत चलन है                                         \n",
              "87     वो मतलबी है अपने बारे में सोचता है                                        \n",
              "202    आवारा बैल खेत में जाके रुकते हैं                                          \n",
              "74     घर अस्तव्यस्त पड़ा है                                                     \n",
              "214    अब भेड़िए नहीं दिखते                                                      \n",
              "210    अपने हितौशी को कुत्ते भी जानते है                                         \n",
              "121    मैं खाता हूँ                                                              \n",
              "255    माँ रोटियाँ सेंक रही है                                                   \n",
              "20     धूप में फूल मुरझा जाता है                                                 \n",
              "188    पांडवों ने कौरव सेना को हराया दिया                                        \n",
              "71     इतना मत खाओ                                                               \n",
              "106    भजिया इमली की चटनी के साथ बहुत ही स्वादिष्ट लगती है                       \n",
              "14     चावल डालें                                                                \n",
              "92     अभी फिर से जाने के लिए कह रहे हैं                                         \n",
              "179    किसी को कभी बिना सोचे समझे नहीं कहना चाहिए दुख होता है                    \n",
              "102    आजकल बारो आमरस मिलता है                                                   \n",
              "Name: hindi_sentance, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "len(X_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jy96GFsc140k",
        "outputId": "cf9c3c0a-a0ae-436a-9cb3-1cf5d035bffa"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "52"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def generate_batch(X = X_train, y = y_train, batch_size = 64):\n",
        "    ''' Generate a batch of data '''\n",
        "    while True:\n",
        "        for j in range(0, len(X), batch_size):\n",
        "            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')\n",
        "            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')\n",
        "            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')\n",
        "            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):\n",
        "                for t, word in enumerate(input_text.split()):\n",
        "                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq\n",
        "                for t, word in enumerate(target_text.split()):\n",
        "                    if t<len(target_text.split())-1:\n",
        "                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq\n",
        "                    if t>0:\n",
        "                        # decoder target sequence (one hot encoded)\n",
        "                        # does not include the START_ token\n",
        "                        # Offset by one timestep\n",
        "                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.\n",
        "            yield([encoder_input_data, decoder_input_data], decoder_target_data)"
      ],
      "metadata": {
        "id": "S7LMs3stnSzG"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "latent_dim = 150\n",
        "# Encoder\n",
        "encoder_inputs = Input(shape=(None,))\n",
        "enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)\n",
        "encoder_lstm = LSTM(latent_dim, return_state=True)\n",
        "encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)\n",
        "# We discard `encoder_outputs` and only keep the states.\n",
        "encoder_states = [state_h, state_c]"
      ],
      "metadata": {
        "id": "EM6ysARHnS2v"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Set up the decoder, using `encoder_states` as initial state.\n",
        "decoder_inputs = Input(shape=(None,))\n",
        "dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)\n",
        "dec_emb = dec_emb_layer(decoder_inputs)\n",
        "# We set up our decoder to return full output sequences,\n",
        "# and to return internal states as well. We don't use the\n",
        "# return states in the training model, but we will use them in inference.\n",
        "decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)\n",
        "decoder_outputs, _, _ = decoder_lstm(dec_emb,\n",
        "                                     initial_state=encoder_states)\n",
        "decoder_dense = Dense(num_decoder_tokens, activation='softmax')\n",
        "decoder_outputs = decoder_dense(decoder_outputs)\n",
        "\n",
        "# Define the model that will turn\n",
        "# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`\n",
        "model = Model([encoder_inputs, decoder_inputs], decoder_outputs)"
      ],
      "metadata": {
        "id": "hmUP5Tk7nS6Z"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "TSMZUh8mnS99"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()\n",
        "train_samples = len(X_train)\n",
        "val_samples = len(X_test)\n",
        "batch_size = 64\n",
        "epochs = 200"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "K4ioUowlnTBc",
        "outputId": "28632c31-55c8-45e5-e256-61801c30fc29"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model\"\n",
            "__________________________________________________________________________________________________\n",
            " Layer (type)                Output Shape                 Param #   Connected to                  \n",
            "==================================================================================================\n",
            " input_1 (InputLayer)        [(None, None)]               0         []                            \n",
            "                                                                                                  \n",
            " input_2 (InputLayer)        [(None, None)]               0         []                            \n",
            "                                                                                                  \n",
            " embedding (Embedding)       (None, None, 150)            112350    ['input_1[0][0]']             \n",
            "                                                                                                  \n",
            " embedding_1 (Embedding)     (None, None, 150)            119100    ['input_2[0][0]']             \n",
            "                                                                                                  \n",
            " lstm (LSTM)                 [(None, 150),                180600    ['embedding[0][0]']           \n",
            "                              (None, 150),                                                        \n",
            "                              (None, 150)]                                                        \n",
            "                                                                                                  \n",
            " lstm_1 (LSTM)               [(None, None, 150),          180600    ['embedding_1[0][0]',         \n",
            "                              (None, 150),                           'lstm[0][1]',                \n",
            "                              (None, 150)]                           'lstm[0][2]']                \n",
            "                                                                                                  \n",
            " dense (Dense)               (None, None, 794)            119894    ['lstm_1[0][0]']              \n",
            "                                                                                                  \n",
            "==================================================================================================\n",
            "Total params: 712544 (2.72 MB)\n",
            "Trainable params: 712544 (2.72 MB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "__________________________________________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),\n",
        "                    steps_per_epoch = train_samples//batch_size,\n",
        "                    epochs=200,\n",
        "                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),\n",
        "                    validation_steps = val_samples//batch_size)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "waF-AxuonTE-",
        "outputId": "2c816fd2-466a-413e-8603-19645e8a7a14"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/200\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-2274d586cca0>:1: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3/3 [==============================] - 11s 164ms/step - loss: 6.6752 - accuracy: 0.0546\n",
            "Epoch 2/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 6.6635 - accuracy: 0.1472\n",
            "Epoch 3/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 6.6431 - accuracy: 0.1472\n",
            "Epoch 4/200\n",
            "3/3 [==============================] - 1s 209ms/step - loss: 6.5961 - accuracy: 0.1466\n",
            "Epoch 5/200\n",
            "3/3 [==============================] - 1s 225ms/step - loss: 6.4715 - accuracy: 0.1456\n",
            "Epoch 6/200\n",
            "3/3 [==============================] - 1s 221ms/step - loss: 6.1287 - accuracy: 0.1472\n",
            "Epoch 7/200\n",
            "3/3 [==============================] - 1s 235ms/step - loss: 5.7406 - accuracy: 0.1472\n",
            "Epoch 8/200\n",
            "3/3 [==============================] - 1s 147ms/step - loss: 5.5492 - accuracy: 0.1466\n",
            "Epoch 9/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 5.6653 - accuracy: 0.1456\n",
            "Epoch 10/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 5.3640 - accuracy: 0.1472\n",
            "Epoch 11/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 5.3226 - accuracy: 0.1472\n",
            "Epoch 12/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 5.2842 - accuracy: 0.1518\n",
            "Epoch 13/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 5.4972 - accuracy: 0.1509\n",
            "Epoch 14/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 5.2007 - accuracy: 0.1493\n",
            "Epoch 15/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 5.1585 - accuracy: 0.1472\n",
            "Epoch 16/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 5.1258 - accuracy: 0.1466\n",
            "Epoch 17/200\n",
            "3/3 [==============================] - 0s 129ms/step - loss: 5.3855 - accuracy: 0.1501\n",
            "Epoch 18/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 5.0633 - accuracy: 0.1567\n",
            "Epoch 19/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 5.0286 - accuracy: 0.1588\n",
            "Epoch 20/200\n",
            "3/3 [==============================] - 0s 126ms/step - loss: 5.0009 - accuracy: 0.1602\n",
            "Epoch 21/200\n",
            "3/3 [==============================] - 0s 155ms/step - loss: 5.2861 - accuracy: 0.1622\n",
            "Epoch 22/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 4.9480 - accuracy: 0.1672\n",
            "Epoch 23/200\n",
            "3/3 [==============================] - 0s 146ms/step - loss: 4.9164 - accuracy: 0.1651\n",
            "Epoch 24/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 4.8819 - accuracy: 0.1686\n",
            "Epoch 25/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 5.1764 - accuracy: 0.1698\n",
            "Epoch 26/200\n",
            "3/3 [==============================] - 0s 147ms/step - loss: 4.8354 - accuracy: 0.1735\n",
            "Epoch 27/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 4.8050 - accuracy: 0.1703\n",
            "Epoch 28/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 4.7700 - accuracy: 0.1717\n",
            "Epoch 29/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 5.0690 - accuracy: 0.1729\n",
            "Epoch 30/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 4.7250 - accuracy: 0.1756\n",
            "Epoch 31/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 4.6988 - accuracy: 0.1703\n",
            "Epoch 32/200\n",
            "3/3 [==============================] - 0s 145ms/step - loss: 4.6596 - accuracy: 0.1749\n",
            "Epoch 33/200\n",
            "3/3 [==============================] - 1s 246ms/step - loss: 4.9642 - accuracy: 0.1751\n",
            "Epoch 34/200\n",
            "3/3 [==============================] - 1s 250ms/step - loss: 4.6115 - accuracy: 0.1798\n",
            "Epoch 35/200\n",
            "3/3 [==============================] - 1s 251ms/step - loss: 4.5850 - accuracy: 0.1756\n",
            "Epoch 36/200\n",
            "3/3 [==============================] - 1s 192ms/step - loss: 4.5453 - accuracy: 0.1822\n",
            "Epoch 37/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 4.8555 - accuracy: 0.1804\n",
            "Epoch 38/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 4.4924 - accuracy: 0.1830\n",
            "Epoch 39/200\n",
            "3/3 [==============================] - 0s 130ms/step - loss: 4.4664 - accuracy: 0.1798\n",
            "Epoch 40/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 4.4236 - accuracy: 0.1812\n",
            "Epoch 41/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 4.7427 - accuracy: 0.1827\n",
            "Epoch 42/200\n",
            "3/3 [==============================] - 0s 141ms/step - loss: 4.3674 - accuracy: 0.1903\n",
            "Epoch 43/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 4.3413 - accuracy: 0.1851\n",
            "Epoch 44/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 4.2968 - accuracy: 0.1906\n",
            "Epoch 45/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 4.6271 - accuracy: 0.1888\n",
            "Epoch 46/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 4.2380 - accuracy: 0.1945\n",
            "Epoch 47/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 4.2119 - accuracy: 0.1851\n",
            "Epoch 48/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 4.1670 - accuracy: 0.1948\n",
            "Epoch 49/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 4.5090 - accuracy: 0.1895\n",
            "Epoch 50/200\n",
            "3/3 [==============================] - 0s 123ms/step - loss: 4.1061 - accuracy: 0.1977\n",
            "Epoch 51/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 4.0788 - accuracy: 0.1893\n",
            "Epoch 52/200\n",
            "3/3 [==============================] - 0s 126ms/step - loss: 4.0348 - accuracy: 0.1969\n",
            "Epoch 53/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 4.3894 - accuracy: 0.1941\n",
            "Epoch 54/200\n",
            "3/3 [==============================] - 0s 147ms/step - loss: 3.9719 - accuracy: 0.1998\n",
            "Epoch 55/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 3.9433 - accuracy: 0.1924\n",
            "Epoch 56/200\n",
            "3/3 [==============================] - 0s 134ms/step - loss: 3.9007 - accuracy: 0.2031\n",
            "Epoch 57/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 4.2714 - accuracy: 0.2024\n",
            "Epoch 58/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 3.8365 - accuracy: 0.2103\n",
            "Epoch 59/200\n",
            "3/3 [==============================] - 0s 141ms/step - loss: 3.8075 - accuracy: 0.2029\n",
            "Epoch 60/200\n",
            "3/3 [==============================] - 0s 141ms/step - loss: 3.7661 - accuracy: 0.2157\n",
            "Epoch 61/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 4.1543 - accuracy: 0.2123\n",
            "Epoch 62/200\n",
            "3/3 [==============================] - 1s 226ms/step - loss: 3.7016 - accuracy: 0.2261\n",
            "Epoch 63/200\n",
            "3/3 [==============================] - 1s 231ms/step - loss: 3.6731 - accuracy: 0.2145\n",
            "Epoch 64/200\n",
            "3/3 [==============================] - 1s 244ms/step - loss: 3.6320 - accuracy: 0.2283\n",
            "Epoch 65/200\n",
            "3/3 [==============================] - 1s 214ms/step - loss: 4.0454 - accuracy: 0.2176\n",
            "Epoch 66/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 3.5817 - accuracy: 0.2303\n",
            "Epoch 67/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 3.5540 - accuracy: 0.2208\n",
            "Epoch 68/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 3.5072 - accuracy: 0.2356\n",
            "Epoch 69/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 3.9306 - accuracy: 0.2320\n",
            "Epoch 70/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 3.4444 - accuracy: 0.2440\n",
            "Epoch 71/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 3.4127 - accuracy: 0.2334\n",
            "Epoch 72/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 3.3692 - accuracy: 0.2545\n",
            "Epoch 73/200\n",
            "3/3 [==============================] - 0s 132ms/step - loss: 3.8091 - accuracy: 0.2456\n",
            "Epoch 74/200\n",
            "3/3 [==============================] - 0s 132ms/step - loss: 3.3025 - accuracy: 0.2524\n",
            "Epoch 75/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 3.2733 - accuracy: 0.2576\n",
            "Epoch 76/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 3.2333 - accuracy: 0.2723\n",
            "Epoch 77/200\n",
            "3/3 [==============================] - 0s 134ms/step - loss: 3.6920 - accuracy: 0.2540\n",
            "Epoch 78/200\n",
            "3/3 [==============================] - 0s 122ms/step - loss: 3.1668 - accuracy: 0.2723\n",
            "Epoch 79/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 3.1296 - accuracy: 0.2702\n",
            "Epoch 80/200\n",
            "3/3 [==============================] - 0s 146ms/step - loss: 3.0882 - accuracy: 0.2953\n",
            "Epoch 81/200\n",
            "3/3 [==============================] - 1s 223ms/step - loss: 3.5540 - accuracy: 0.2707\n",
            "Epoch 82/200\n",
            "3/3 [==============================] - 1s 247ms/step - loss: 3.0161 - accuracy: 0.2871\n",
            "Epoch 83/200\n",
            "3/3 [==============================] - 1s 239ms/step - loss: 2.9799 - accuracy: 0.3028\n",
            "Epoch 84/200\n",
            "3/3 [==============================] - 1s 188ms/step - loss: 2.9398 - accuracy: 0.3215\n",
            "Epoch 85/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 3.4196 - accuracy: 0.2889\n",
            "Epoch 86/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 2.8707 - accuracy: 0.3113\n",
            "Epoch 87/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 2.8312 - accuracy: 0.3323\n",
            "Epoch 88/200\n",
            "3/3 [==============================] - 1s 373ms/step - loss: 2.7918 - accuracy: 0.3487\n",
            "Epoch 89/200\n",
            "3/3 [==============================] - 1s 340ms/step - loss: 3.2846 - accuracy: 0.3184\n",
            "Epoch 90/200\n",
            "3/3 [==============================] - 1s 242ms/step - loss: 2.7279 - accuracy: 0.3396\n",
            "Epoch 91/200\n",
            "3/3 [==============================] - 1s 153ms/step - loss: 2.6898 - accuracy: 0.3638\n",
            "Epoch 92/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 2.6565 - accuracy: 0.3801\n",
            "Epoch 93/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 3.1494 - accuracy: 0.3389\n",
            "Epoch 94/200\n",
            "3/3 [==============================] - 0s 126ms/step - loss: 2.5880 - accuracy: 0.3743\n",
            "Epoch 95/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 2.5410 - accuracy: 0.3975\n",
            "Epoch 96/200\n",
            "3/3 [==============================] - 0s 145ms/step - loss: 2.5074 - accuracy: 0.4084\n",
            "Epoch 97/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 3.0036 - accuracy: 0.3707\n",
            "Epoch 98/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 2.4450 - accuracy: 0.3996\n",
            "Epoch 99/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 2.3962 - accuracy: 0.4269\n",
            "Epoch 100/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 2.3619 - accuracy: 0.4346\n",
            "Epoch 101/200\n",
            "3/3 [==============================] - 0s 145ms/step - loss: 2.8577 - accuracy: 0.3958\n",
            "Epoch 102/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 2.2923 - accuracy: 0.4290\n",
            "Epoch 103/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 2.2443 - accuracy: 0.4490\n",
            "Epoch 104/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 2.2120 - accuracy: 0.4607\n",
            "Epoch 105/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 2.7049 - accuracy: 0.4291\n",
            "Epoch 106/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 2.1549 - accuracy: 0.4637\n",
            "Epoch 107/200\n",
            "3/3 [==============================] - 0s 126ms/step - loss: 2.1048 - accuracy: 0.4848\n",
            "Epoch 108/200\n",
            "3/3 [==============================] - 0s 132ms/step - loss: 2.0712 - accuracy: 0.5037\n",
            "Epoch 109/200\n",
            "3/3 [==============================] - 1s 334ms/step - loss: 2.5562 - accuracy: 0.4678\n",
            "Epoch 110/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 2.0167 - accuracy: 0.5058\n",
            "Epoch 111/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 1.9687 - accuracy: 0.5205\n",
            "Epoch 112/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 1.9374 - accuracy: 0.5424\n",
            "Epoch 113/200\n",
            "3/3 [==============================] - 0s 148ms/step - loss: 2.4134 - accuracy: 0.4958\n",
            "Epoch 114/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 1.8902 - accuracy: 0.5415\n",
            "Epoch 115/200\n",
            "3/3 [==============================] - 1s 219ms/step - loss: 1.8450 - accuracy: 0.5489\n",
            "Epoch 116/200\n",
            "3/3 [==============================] - 1s 249ms/step - loss: 1.8164 - accuracy: 0.5853\n",
            "Epoch 117/200\n",
            "3/3 [==============================] - 1s 236ms/step - loss: 2.2824 - accuracy: 0.5534\n",
            "Epoch 118/200\n",
            "3/3 [==============================] - 1s 231ms/step - loss: 1.7768 - accuracy: 0.5731\n",
            "Epoch 119/200\n",
            "3/3 [==============================] - 1s 161ms/step - loss: 1.7268 - accuracy: 0.5857\n",
            "Epoch 120/200\n",
            "3/3 [==============================] - 0s 148ms/step - loss: 1.6925 - accuracy: 0.6251\n",
            "Epoch 121/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 2.1357 - accuracy: 0.5792\n",
            "Epoch 122/200\n",
            "3/3 [==============================] - 0s 141ms/step - loss: 1.6544 - accuracy: 0.6278\n",
            "Epoch 123/200\n",
            "3/3 [==============================] - 0s 147ms/step - loss: 1.6020 - accuracy: 0.6341\n",
            "Epoch 124/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 1.5693 - accuracy: 0.6545\n",
            "Epoch 125/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 1.9981 - accuracy: 0.6391\n",
            "Epoch 126/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 1.5406 - accuracy: 0.6551\n",
            "Epoch 127/200\n",
            "3/3 [==============================] - 0s 139ms/step - loss: 1.4913 - accuracy: 0.6877\n",
            "Epoch 128/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 1.4592 - accuracy: 0.7120\n",
            "Epoch 129/200\n",
            "3/3 [==============================] - 0s 126ms/step - loss: 1.8747 - accuracy: 0.6755\n",
            "Epoch 130/200\n",
            "3/3 [==============================] - 0s 145ms/step - loss: 1.4445 - accuracy: 0.7045\n",
            "Epoch 131/200\n",
            "3/3 [==============================] - 0s 125ms/step - loss: 1.3838 - accuracy: 0.7329\n",
            "Epoch 132/200\n",
            "3/3 [==============================] - 0s 129ms/step - loss: 1.3718 - accuracy: 0.7340\n",
            "Epoch 133/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 1.7337 - accuracy: 0.7377\n",
            "Epoch 134/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 1.3190 - accuracy: 0.7487\n",
            "Epoch 135/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 1.2683 - accuracy: 0.7708\n",
            "Epoch 136/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 1.2411 - accuracy: 0.7843\n",
            "Epoch 137/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 1.5927 - accuracy: 0.7771\n",
            "Epoch 138/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 1.2110 - accuracy: 0.7876\n",
            "Epoch 139/200\n",
            "3/3 [==============================] - 0s 124ms/step - loss: 1.1620 - accuracy: 0.8202\n",
            "Epoch 140/200\n",
            "3/3 [==============================] - 0s 134ms/step - loss: 1.1364 - accuracy: 0.8262\n",
            "Epoch 141/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 1.4641 - accuracy: 0.8264\n",
            "Epoch 142/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 1.1134 - accuracy: 0.8318\n",
            "Epoch 143/200\n",
            "3/3 [==============================] - 0s 144ms/step - loss: 1.0665 - accuracy: 0.8591\n",
            "Epoch 144/200\n",
            "3/3 [==============================] - 1s 206ms/step - loss: 1.0414 - accuracy: 0.8681\n",
            "Epoch 145/200\n",
            "3/3 [==============================] - 1s 229ms/step - loss: 1.3472 - accuracy: 0.8582\n",
            "Epoch 146/200\n",
            "3/3 [==============================] - 1s 231ms/step - loss: 1.0250 - accuracy: 0.8580\n",
            "Epoch 147/200\n",
            "3/3 [==============================] - 1s 227ms/step - loss: 0.9794 - accuracy: 0.8875\n",
            "Epoch 148/200\n",
            "3/3 [==============================] - 1s 181ms/step - loss: 0.9546 - accuracy: 0.8911\n",
            "Epoch 149/200\n",
            "3/3 [==============================] - 0s 134ms/step - loss: 1.2393 - accuracy: 0.8802\n",
            "Epoch 150/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.9430 - accuracy: 0.8812\n",
            "Epoch 151/200\n",
            "3/3 [==============================] - 0s 148ms/step - loss: 0.8996 - accuracy: 0.9022\n",
            "Epoch 152/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 0.8736 - accuracy: 0.9099\n",
            "Epoch 153/200\n",
            "3/3 [==============================] - 0s 151ms/step - loss: 1.1391 - accuracy: 0.8954\n",
            "Epoch 154/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.8709 - accuracy: 0.9001\n",
            "Epoch 155/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.8264 - accuracy: 0.9159\n",
            "Epoch 156/200\n",
            "3/3 [==============================] - 0s 153ms/step - loss: 0.8028 - accuracy: 0.9267\n",
            "Epoch 157/200\n",
            "3/3 [==============================] - 0s 129ms/step - loss: 1.0506 - accuracy: 0.9151\n",
            "Epoch 158/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 0.7996 - accuracy: 0.9159\n",
            "Epoch 159/200\n",
            "3/3 [==============================] - 0s 129ms/step - loss: 0.7575 - accuracy: 0.9327\n",
            "Epoch 160/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 0.7357 - accuracy: 0.9424\n",
            "Epoch 161/200\n",
            "3/3 [==============================] - 0s 147ms/step - loss: 0.9545 - accuracy: 0.9340\n",
            "Epoch 162/200\n",
            "3/3 [==============================] - 0s 127ms/step - loss: 0.7265 - accuracy: 0.9390\n",
            "Epoch 163/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 0.6904 - accuracy: 0.9485\n",
            "Epoch 164/200\n",
            "3/3 [==============================] - 0s 131ms/step - loss: 0.6660 - accuracy: 0.9560\n",
            "Epoch 165/200\n",
            "3/3 [==============================] - 0s 145ms/step - loss: 0.8695 - accuracy: 0.9507\n",
            "Epoch 166/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 0.6639 - accuracy: 0.9516\n",
            "Epoch 167/200\n",
            "3/3 [==============================] - 0s 132ms/step - loss: 0.6296 - accuracy: 0.9558\n",
            "Epoch 168/200\n",
            "3/3 [==============================] - 0s 136ms/step - loss: 0.6070 - accuracy: 0.9665\n",
            "Epoch 169/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 0.7925 - accuracy: 0.9613\n",
            "Epoch 170/200\n",
            "3/3 [==============================] - 0s 146ms/step - loss: 0.6070 - accuracy: 0.9611\n",
            "Epoch 171/200\n",
            "3/3 [==============================] - 0s 153ms/step - loss: 0.5756 - accuracy: 0.9642\n",
            "Epoch 172/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 0.5532 - accuracy: 0.9728\n",
            "Epoch 173/200\n",
            "3/3 [==============================] - 1s 231ms/step - loss: 0.7240 - accuracy: 0.9682\n",
            "Epoch 174/200\n",
            "3/3 [==============================] - 1s 253ms/step - loss: 0.5560 - accuracy: 0.9685\n",
            "Epoch 175/200\n",
            "3/3 [==============================] - 1s 255ms/step - loss: 0.5268 - accuracy: 0.9727\n",
            "Epoch 176/200\n",
            "3/3 [==============================] - 1s 194ms/step - loss: 0.5055 - accuracy: 0.9759\n",
            "Epoch 177/200\n",
            "3/3 [==============================] - 0s 138ms/step - loss: 0.6613 - accuracy: 0.9712\n",
            "Epoch 178/200\n",
            "3/3 [==============================] - 0s 140ms/step - loss: 0.5090 - accuracy: 0.9737\n",
            "Epoch 179/200\n",
            "3/3 [==============================] - 0s 139ms/step - loss: 0.4826 - accuracy: 0.9769\n",
            "Epoch 180/200\n",
            "3/3 [==============================] - 0s 149ms/step - loss: 0.4619 - accuracy: 0.9822\n",
            "Epoch 181/200\n",
            "3/3 [==============================] - 0s 148ms/step - loss: 0.6048 - accuracy: 0.9773\n",
            "Epoch 182/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.4674 - accuracy: 0.9758\n",
            "Epoch 183/200\n",
            "3/3 [==============================] - 0s 148ms/step - loss: 0.4428 - accuracy: 0.9779\n",
            "Epoch 184/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.4232 - accuracy: 0.9832\n",
            "Epoch 185/200\n",
            "3/3 [==============================] - 0s 139ms/step - loss: 0.5538 - accuracy: 0.9795\n",
            "Epoch 186/200\n",
            "3/3 [==============================] - 0s 147ms/step - loss: 0.4287 - accuracy: 0.9821\n",
            "Epoch 187/200\n",
            "3/3 [==============================] - 1s 245ms/step - loss: 0.4066 - accuracy: 0.9832\n",
            "Epoch 188/200\n",
            "3/3 [==============================] - 0s 142ms/step - loss: 0.3883 - accuracy: 0.9885\n",
            "Epoch 189/200\n",
            "3/3 [==============================] - 1s 200ms/step - loss: 0.5080 - accuracy: 0.9848\n",
            "Epoch 190/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 0.3944 - accuracy: 0.9821\n",
            "Epoch 191/200\n",
            "3/3 [==============================] - 1s 154ms/step - loss: 0.3742 - accuracy: 0.9853\n",
            "Epoch 192/200\n",
            "3/3 [==============================] - 0s 137ms/step - loss: 0.3571 - accuracy: 0.9906\n",
            "Epoch 193/200\n",
            "3/3 [==============================] - 0s 128ms/step - loss: 0.4666 - accuracy: 0.9856\n",
            "Epoch 194/200\n",
            "3/3 [==============================] - 0s 133ms/step - loss: 0.3630 - accuracy: 0.9853\n",
            "Epoch 195/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.3451 - accuracy: 0.9884\n",
            "Epoch 196/200\n",
            "3/3 [==============================] - 0s 143ms/step - loss: 0.3297 - accuracy: 0.9895\n",
            "Epoch 197/200\n",
            "3/3 [==============================] - 0s 151ms/step - loss: 0.4304 - accuracy: 0.9886\n",
            "Epoch 198/200\n",
            "3/3 [==============================] - 0s 135ms/step - loss: 0.3349 - accuracy: 0.9895\n",
            "Epoch 199/200\n",
            "3/3 [==============================] - 1s 234ms/step - loss: 0.3186 - accuracy: 0.9884\n",
            "Epoch 200/200\n",
            "3/3 [==============================] - 1s 226ms/step - loss: 0.3040 - accuracy: 0.9927\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7e0fa235d4e0>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('hindi_to_cg.h5')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bdixtpq2nTIu",
        "outputId": "d5c3c8c5-d4ff-46cd-c441-dc154ce3d786"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_gen = generate_batch(X_train, y_train, batch_size = 1)\n",
        "k=-1"
      ],
      "metadata": {
        "id": "zoL71Pw-nTMZ"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Encode the input sequence to get the \"thought vectors\"\n",
        "encoder_model = Model(encoder_inputs, encoder_states)\n",
        "\n",
        "# Decoder setup\n",
        "# Below tensors will hold the states of the previous time step\n",
        "decoder_state_input_h = Input(shape=(latent_dim,))\n",
        "decoder_state_input_c = Input(shape=(latent_dim,))\n",
        "decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]\n",
        "\n",
        "dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence\n",
        "\n",
        "# To predict the next word in the sequence, set the initial states to the states from the previous time step\n",
        "decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)\n",
        "decoder_states2 = [state_h2, state_c2]\n",
        "decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary\n",
        "\n",
        "# Final decoder model\n",
        "decoder_model = Model(\n",
        "    [decoder_inputs] + decoder_states_inputs,\n",
        "    [decoder_outputs2] + decoder_states2)"
      ],
      "metadata": {
        "id": "hsrh5RSonTQC"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def decode_sequence(input_seq):\n",
        "    # Encode the input as state vectors.\n",
        "    states_value = encoder_model.predict(input_seq)\n",
        "    # Generate empty target sequence of length 1.\n",
        "    target_seq = np.zeros((1,1))\n",
        "    # Populate the first character of target sequence with the start character.\n",
        "    target_seq[0, 0] = target_token_index['START_']\n",
        "\n",
        "    # Sampling loop for a batch of sequences\n",
        "    # (to simplify, here we assume a batch of size 1).\n",
        "    stop_condition = False\n",
        "    decoded_sentence = ''\n",
        "    while not stop_condition:\n",
        "        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)\n",
        "\n",
        "        # Sample a token\n",
        "        sampled_token_index = np.argmax(output_tokens[0, -1, :])\n",
        "        sampled_char = reverse_target_char_index[sampled_token_index]\n",
        "        decoded_sentence += ' '+sampled_char\n",
        "\n",
        "        # Exit condition: either hit max length\n",
        "        # or find stop character.\n",
        "        if (sampled_char == '_END' or\n",
        "           len(decoded_sentence) > 50):\n",
        "            stop_condition = True\n",
        "\n",
        "        # Update the target sequence (of length 1).\n",
        "        target_seq = np.zeros((1,1))\n",
        "        target_seq[0, 0] = sampled_token_index\n",
        "\n",
        "        # Update states\n",
        "        states_value = [h, c]\n",
        "\n",
        "    return decoded_sentence"
      ],
      "metadata": {
        "id": "Iz1hjIARnTTs"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "k+=1\n",
        "(input_seq, actual_output), _ = next(train_gen)\n",
        "decoded_sentence = decode_sequence(input_seq)\n",
        "print('Input English sentence:', X_train[k:k+1].values[0])\n",
        "print('Actual cg Translation:', y_train[k:k+1].values[0][6:-4])\n",
        "print('Predicted cg Translation:', decoded_sentence[:-4])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p7MiMJiZnTXN",
        "outputId": "2957ef3d-f74a-47dd-e4bc-198c78c76d57"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 1s 1s/step\n",
            "1/1 [==============================] - 1s 1s/step\n",
            "1/1 [==============================] - 0s 27ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "Input English sentence: तुमने लिखा\n",
            "Actual cg Translation:  तुमन लिखेव \n",
            "Predicted cg Translation:  तें लिखेस \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "k+=1\n",
        "(input_seq, actual_output), _ = next(train_gen)\n",
        "decoded_sentence = decode_sequence(input_seq)\n",
        "print('Input English sentence:', X_train[k:k+1].values[0])\n",
        "print('Actual cg Translation:', y_train[k:k+1].values[0][6:-4])\n",
        "print('Predicted cg Translation:', decoded_sentence[:-4])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AFI_ZtvXnTa3",
        "outputId": "b8c9eda7-437d-4e62-f9d8-10acc10aaf5c"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 21ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 24ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "Input English sentence: हरिया का घर यहीं से सीधा है कहीं और मत जाना\n",
            "Actual cg Translation:  हरिया के घर ईंहा ले सोज्झे हे अंते झन जाबे \n",
            "Predicted cg Translation:  हरिया के घर ईंहा ले सोज्झे हे अंते झन जाबे \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "k+=1\n",
        "(input_seq, actual_output), _ = next(train_gen)\n",
        "decoded_sentence = decode_sequence(input_seq)\n",
        "print('Input English sentence:', X_train[k:k+1].values[0])\n",
        "print('Actual cg Translation:', y_train[k:k+1].values[0][6:-4])\n",
        "print('Predicted cg Translation:', decoded_sentence[:-4])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Lt7C2y1CnTeh",
        "outputId": "17d27248-807e-44aa-d16d-592112c32476"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 28ms/step\n",
            "1/1 [==============================] - 0s 23ms/step\n",
            "1/1 [==============================] - 0s 24ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 24ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "1/1 [==============================] - 0s 22ms/step\n",
            "Input English sentence: मुसीबत में काम आये वही मित्र है\n",
            "Actual cg Translation:  अटके परे म काम आथे तउने मितान हे \n",
            "Predicted cg Translation:  अटके परे म काम आथे तउने मितान हे \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "RzfOLKNPnTiK"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "DB4QJcsJnTl8"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "w7C1ps9dnTp2"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Cy2lOd2KnTt4"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "txmuROZvnTxx"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "BdJTt_6_nT1j"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "DAdoAu-cnT5V"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "7CgQs941nT9X"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "hXXzhew4nUBQ"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "dL02iSuInUFC"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ZBhRcsDEnUIs"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "w-m_YU4znUMV"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "NhES2wpEnUQH"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "7iV8RGo6nUUQ"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Bv2rVAWDnUX6"
      },
      "execution_count": 36,
      "outputs": []
    }
  ]
}
