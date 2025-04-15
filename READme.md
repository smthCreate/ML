docker build -t mlmodel .
docker run -d --name mlmodel-container -p5000:5000 mlmodel
git
test