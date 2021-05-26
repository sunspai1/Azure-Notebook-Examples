FROM 6e64c58548454356b1e0a28ca62f252a.azurecr.io/octopy:latest
COPY . /data
COPY mungerclient /usr/local/bin/
WORKDIR /data

#RUN pip install -r requirements.txt

RUN pip install pandas==1.2.1 \
fastcore==1.3.20 \
numpy==1.19.2 \
tqdm==4.60.0 \
papermill[azure]==2.3.3 \
jupyter==1.0.0 \
azure-storage-blob==12.8.0 \
scikit-learn==0.24.2

RUN chmod u+x run.sh
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1
RUN ls
ENV HOME /data