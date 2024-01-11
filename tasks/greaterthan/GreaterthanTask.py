import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tasks.task import Task

class GreaterthanTask(Task):
    def __init__(self, model, batch_size, device="cuda"):
        self.NOUNS = [
            "abduction", "accord", "affair", "agreement", "appraisal",
            "assaults", "assessment", "attack", "attempts", "campaign", 
            "captivity", "case", "challenge", "chaos", "clash", 
            "collaboration", "coma", "competition", "confrontation", "consequence", 
            "conspiracy", "construction", "consultation", "contact",
            "contract", "convention", "cooperation", "custody", "deal", 
            "decline", "decrease", "demonstrations", "development", "disagreement", 
            "disorder", "dispute", "domination", "dynasty", "effect", 
            "effort", "employment", "endeavor", "engagement",
            "epidemic", "evaluation", "exchange", "existence", "expansion", 
            "expedition", "experiments", "fall", "fame", "flights",
            "friendship", "growth", "hardship", "hostility", "illness", 
            "impact", "imprisonment", "improvement", "incarceration",
            "increase", "insurgency", "invasion", "investigation", "journey", 
            "kingdom", "marriage", "modernization", "negotiation",
            "notoriety", "obstruction", "operation", "order", "outbreak", 
            "outcome", "overhaul", "patrols", "pilgrimage", "plague",
            "plan", "practice", "process", "program", "progress", 
            "project", "pursuit", "quest", "raids", "reforms", 
            "reign", "relationship",
            "retaliation", "riot", "rise", "rivalry", "romance", 
            "rule", "sanctions", "shift", "siege", "slump", 
            "stature", "stint", "strikes", "study",
            "test", "testing", "tests", "therapy", "tour", 
            "tradition", "treaty", "trial", "trip", "unemployment", 
            "voyage", "warfare", "work",
        ]
        _TOKENIZER = model.tokenizer

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [_TOKENIZER.encode(f" {str(year)[:2]}")[0], _TOKENIZER.encode(str(year)[2:])[0]]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {
            i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(0, 100)
        }
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(0, 100)], dtype=torch.long, device=device)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long, device=device)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR

        # Getting data
        data, prompts = self.get_year_data(batch_size*2, model)
        self.train_data = data[:batch_size]
        self.test_data = data[batch_size:]
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

        # Setting up criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def get_year_data(self, num_examples, model):
        template = "The {noun} lasted from the year {year1} to "

        # set some random seed
        torch.random.manual_seed(54)
        nouns_perm = torch.randint(0, len(self.NOUNS), (num_examples,))
        years_perm = torch.randint(0, len(self.YEARS), (num_examples,))

        prompts = []
        prompts_tokenized = []
        for i in range(num_examples):
            year = self.YEARS[years_perm[i]]
            prompts.append(
                template.format(
                    noun=self.NOUNS[nouns_perm[i]],
                    year1=year,
                ) + year[:2]
            )
            prompts_tokenized.append(model.tokenizer.encode(prompts[-1], return_tensors="pt").to(model.cfg.device))
            assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
        prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
        assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

        return prompts_tokenized, prompts

    def compute_means(self,
        model,
        num_data = None
    ):
        """
        Computes the mean of the activations across 
        the data for each component of the model. Used in mean edge ablation.
        """
        raise NotImplementedError
    
    def get_batch(self, train=True):
        """
        Get a batch of data from the task.
        """
        if train:
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
        else:
            try:
                batch = next(self.test_iter)
            except StopIteration:
                self.test_iter = iter(self.test_loader)
                batch = next(self.test_iter)
        return batch

    def calculate_loss(self, model, batch):
        logits = model(batch)
        # Gets the last 2 digits of the year
        yearend = self.INV_TOKENS_TENSOR[batch[:, 7]].to(logits.device)

        # Construct target distribution that is uniform for all years > yearend
        target = torch.zeros_like(logits) # Shape batch, vocab

        # For each year, set the probability of all years after it to 1 / p
        for i in range(len(yearend)):
            p = 100 - yearend[i] + 1
            target[i, self.TOKENS_TENSOR[yearend[i]+1:]] = 1 / p

        target = F.softmax(target, dim=-1)

        # Compute cross entropy loss
        return self.criterion(logits, target)
        
    def get_train_loss(self, model):
        batch = self.get_batch(train=True)
        return self.calculate_loss(model, batch)
    
    def get_test_loss(self, model):
        batch = self.get_batch(train=False)
        with torch.no_grad():
            return self.calculate_loss(model, batch)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        raise NotImplementedError