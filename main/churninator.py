

# Define the neural network architecture
class ChurnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(ChurnNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Churninator():
    def __init__(path_to_file,
                features2use,
                preprocess_file=False,
                network_bestparams=None,
                optimize_optuna=False,
                verbose=False
                
                ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        
        if preprocess_file:
            #... preprocess the excel file, call cleaner
        else:
            df = pd.read_csv(path_to_file, header=0, sep=',')
            
        X,y = df[features2use], df['Exited']
        
        #normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        #make data splits
        X_train, X_val, y_train, y_val = train_test_split(X, 
                                                          y, 
                                                          test_size=0.2, 
                                                          random_state=42, 
                                                          stratify=y)
        X_test, X_val, y_test, y_val = train_test_split(X_val, 
                                                        y_val, 
                                                        test_size=0.1, 
                                                        random_state=42, 
                                                        stratify=y_val)

        #create dataloaders

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train.values.flatten(), dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                    torch.tensor(y_val.values.flatten(), dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        dataloaders = {'train': train_loader, 'val': val_loader}        

        if optimize_optuna:
            optimize_nn_with_optuna #...
        else: 
            self.network_params = 

        self.model = ChurnNet(len(features2use), self.network_params['hidden_dim'], 1, self.network_params['dropout_rate'])

        self._train_model(dataloaders)
        
    
    def _train_model(self, dataloaders):
        self.model = self.model.to(self.device)
        criterion = nn.BCELoss()
        
        if self.network_params['optimizer_name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif self.network_params['optimizer_name'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif self.network_params['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        for epoch in range(self.network_params['num_epochs']):
            if self.verbose:
                print(f'Epoch {epoch}/{num_epochs-1}')
                print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        preds = (outputs > 0.5).float()
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                if self.verbose:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
        return model
                

        
        