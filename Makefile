.PHONY: *

APP_PORT := 5007
CONTAINER_NAME := my-barcode-container
DOCKER_IMAGE := my-barcode-app
DOCKER_TAG := latest

DEPLOY_HOST := demo_host
DVC_REMOTE_NAME := alexander_vedernikov_remote
STAGING_USERNAME := a.vedernikov
KEY_FILE := ~/.ssh/id_rsa

install:
	@bash setup.sh

run_unit_tests:
	PYTHONPATH=. pytest tests/unit/

run_integration_tests:
	PYTHONPATH=. pytest tests/integration/

run_tests:
	make run_unit_tests
	make run_integration_tests

lint:
	PYTHONPATH=. flake8 .

run_app:
	PYTHONPATH=. python3 -m uvicorn app:create_app --host='0.0.0.0' --port=$(APP_PORT) --factory

build_image:
	docker build -f Dockerfile . --force-rm=true -t $(DOCKER_IMAGE):$(DOCKER_TAG)

run_container:
	docker run -d -p $(APP_PORT):$(APP_PORT) --name $(CONTAINER_NAME) $(DOCKER_IMAGE)

stop_container:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

install_dvc:
	pip install pygit2==1.10.1 pathspec==0.9.0
	pip install dvc[ssh]==2.5.4

pull_weights: configure_dvc_remote
	dvc pull -R weights

configure_dvc_remote:
	dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(STAGING_USERNAME)/dvc_files || true
	dvc remote modify $(DVC_REMOTE_NAME) user $(STAGING_USERNAME)
	dvc remote modify $(DVC_REMOTE_NAME) keyfile $(KEY_FILE)
	dvc config cache.type hardlink,symlink

deploy:
	ansible-playbook -vvvv -i deploy/ansible/inventory.ini deploy/ansible/deploy.yml \
		-e host=$(DEPLOY_HOST) \
		-e docker_image=$(DOCKER_IMAGE) \
		-e docker_tag=$(DOCKER_TAG) \
		-e docker_registry_user=$(CI_REGISTRY_USER) \
		-e docker_registry_password=$(CI_REGISTRY_PASSWORD) \
		-e docker_registry=$(CI_REGISTRY)

destroy:
	ansible-playbook -i deploy/ansible/inventory.ini deploy/ansible/destroy.yml \
		-e host=$(DEPLOY_HOST)
