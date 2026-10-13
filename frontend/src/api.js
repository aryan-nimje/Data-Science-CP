import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const checkBackend = async () => {
  const response = await API.get("/");
  return response.data;
};

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const trainModel = async (filename) => {
  const response = await API.post("/train", null, {
    params: {
      filename,
    },
  });

  return response.data;
};

export const getReport = async (filename) => {
  const response = await API.post("/report", {
    filename,
  });

  return response.data;
};

export const predict = async (inputData) => {
  const response = await API.post("/predict", inputData);
  return response.data;
};

export const explainPrediction = async (inputData) => {
  const response = await API.post("/explain", inputData);
  return response.data;
};

export const whatIf = async (
  filename,
  row,
  feature,
  newValue
) => {
  const response = await API.post("/what-if", {
    filename,
    row,
    feature,
    new_value: newValue,
  });

  return response.data;
};
export const getErrors = async (filename) => {
  const response = await API.post("/errors", null, {
    params: {
      filename,
    },
  });

  return response.data;
};

export default API;