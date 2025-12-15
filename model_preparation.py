ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, ipg_handler], deepspeed_plugin=deepspeed_plugin)

warnings.simplefilter(action='ignore', category=FutureWarning)

with open('mllama3.1_train_rcab_az_config.json', 'r') as f:
    configs = json.load(f)
    configs = SimpleNamespace(**configs)

seed = configs.seed   # random.randint(1, 9999)
accelerator.print(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

train_loader = data_provider(configs, 'train')
vali_loader = data_provider(configs, 'val')

path = './checkpoints'

model = ft_timeRAG_fast.Model(configs).float()
model.load_state_dict(torch.load(path + '/checkpoint_tok_rcab.pth',
                                 weights_only=True, map_location='cpu'), strict=False)

gpt_token_provider = get_bearer_token_provider(
    ClientSecretCredential(),
    "https://cognitiveservices.azure.com/.default"
)
gpt_client = AzureOpenAI(
    api_version="2024-10-01-preview",
    azure_endpoint="https://openai4research.openai.azure.com/",
    azure_ad_token_provider=gpt_token_provider,
)

time_now = time.time()

train_steps = len(train_loader)
early_stopping = EarlyStopping(accelerator=accelerator, patience=configs.patience)

trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

model_optim = optim.Adam(trained_parameters, lr=configs.learning_rate)

if configs.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
else:
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=configs.pct_start,
                                        epochs=configs.train_epochs,
                                        max_lr=configs.learning_rate)

criterion = retrival_evaluator(label_name=configs.label_dir)
mse_loss = nn.MSELoss()

train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
    train_loader, vali_loader, model, model_optim, scheduler)
